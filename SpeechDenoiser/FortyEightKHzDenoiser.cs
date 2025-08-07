using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;
using Tensor = TorchSharp.torch.Tensor;

class FortyEightKHzDenoiser
{
    public static void Main(string[] args)
    {
        // Initialize model
        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            IntraOpNumThreads = 1,
            InterOpNumThreads = 1,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
        };

        var ortSession = new InferenceSession(
            "./denoiser_model.onnx",
            sessionOptions
        );

        // Load wav file
        const int hopSize = 480;
        const int fftSize = 960;

        // Load audio using NAudio
        var (inputAudio, sampleRate) = LoadAudioFile("./inp.wav");

        // Convert to mono if stereo
        if (inputAudio.shape[0] > 1)
        {
            inputAudio = inputAudio.mean().unsqueeze(0);
        }
        inputAudio = inputAudio.squeeze(0);

        int origLen = (int)inputAudio.shape[0];

        // Padding calculation
        int hopSizeDivisiblePaddingSize = (hopSize - origLen % hopSize) % hopSize;
        origLen += hopSizeDivisiblePaddingSize;

        // Apply padding
        inputAudio = pad(inputAudio, new long[] { 0, fftSize + hopSizeDivisiblePaddingSize });

        // Split into chunks
        var chunkedAudio = inputAudio.split(hopSize, dim: 0);

        // Initialize state arrays
        var state = new float[45304];
        var attenLimDb = new float[1];
        var enhanced = new List<Tensor>();

        // Inference loop
        foreach (var frame in chunkedAudio)
        {
            // Prepare inputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_frame",
                    new DenseTensor<float>(frame.data<float>().ToArray(),
                    new int[] { (int)frame.shape[0] })),
                NamedOnnxValue.CreateFromTensor("states",
                    new DenseTensor<float>(state, new int[] { state.Length })),
                NamedOnnxValue.CreateFromTensor("atten_lim_db",
                    new DenseTensor<float>(attenLimDb, new int[] { 1 }))
            };

            // Run inference
            using var results = ortSession.Run(inputs);
            var outputList = results.ToList();

            // Extract outputs
            var enhancedFrame = outputList[0].AsTensor<float>().ToArray();
            enhanced.Add(tensor(enhancedFrame));

            // Update state
            state = [.. outputList[1].AsTensor<float>()];
        }

        // Concatenate enhanced audio
        var enhancedAudio = cat([.. enhanced]).unsqueeze(0);

        // Remove padding
        int d = fftSize - hopSize;
        enhancedAudio = enhancedAudio[.., d..(origLen + d)];

        // Save output
        SaveAudioFile("out.wav", enhancedAudio, sampleRate);

        Console.WriteLine("Audio denoising completed. Output saved to out.wav");

        // Cleanup
        ortSession.Dispose();
    }

    static (Tensor audio, int sampleRate) LoadAudioFile(string filePath)
    {
        using var reader = new AudioFileReader(filePath);
        var sampleRate = reader.WaveFormat.SampleRate;
        var channels = reader.WaveFormat.Channels;

        // Read all samples
        var samples = new List<float>();
        var buffer = new float[1024];
        int samplesRead;

        while ((samplesRead = reader.Read(buffer, 0, buffer.Length)) > 0)
        {
            for (int i = 0; i < samplesRead; i++)
            {
                samples.Add(buffer[i]);
            }
        }

        // Convert to tensor
        var audioArray = samples.ToArray();
        Tensor audioTensor;

        if (channels == 1)
        {
            audioTensor = tensor(audioArray).unsqueeze(0);
        }
        else
        {
            // Reshape for multi-channel audio
            var reshapedAudio = new float[channels, audioArray.Length / channels];
            for (int i = 0; i < audioArray.Length; i++)
            {
                reshapedAudio[i % channels, i / channels] = audioArray[i];
            }
            audioTensor = tensor(reshapedAudio);
        }

        return (audioTensor, sampleRate);
    }

    static void SaveAudioFile(string filePath, Tensor audio, int sampleRate)
    {
        // Convert tensor to float array
        var audioData = audio.squeeze(0).data<float>().ToArray();

        // Create wave format
        var waveFormat = new WaveFormat(sampleRate, 16, 1); // 16-bit mono

        // Convert float to 16-bit PCM
        var pcmData = new short[audioData.Length];
        for (int i = 0; i < audioData.Length; i++)
        {
            // Clamp to [-1, 1] and convert to 16-bit
            var sample = Math.Max(-1.0f, Math.Min(1.0f, audioData[i]));
            pcmData[i] = (short)(sample * short.MaxValue);
        }

        // Save as WAV file
        using var writer = new WaveFileWriter(filePath, waveFormat);

        // Convert short array to byte array
        var byteData = new byte[pcmData.Length * 2];
        Buffer.BlockCopy(pcmData, 0, byteData, 0, byteData.Length);

        writer.Write(byteData, 0, byteData.Length);
    }
}