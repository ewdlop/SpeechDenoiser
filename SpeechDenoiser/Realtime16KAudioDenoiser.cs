using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.CoreAudioApi;
using NAudio.Wave;
using TorchSharp;
using static TorchSharp.torch;

class Realtime16KAudioDenoiser
{
    const int SampleRate = 16000;
    const int NFFT = 512;
    const int HopLength = 256;
    const int WinLength = 512;

    static InferenceSession session;
    static float[,,,,] conv_cache = new float[2, 1, 16, 16, 33];
    static float[,,,,] tra_cache = new float[2, 3, 1, 1, 16];
    static float[,,,] inter_cache = new float[2, 1, 33, 16];

    public static void Main(string[] args)
    {
        session = new InferenceSession("gtcrn_simple.onnx");

        var window = torch.hann_window(NFFT, dtype: ScalarType.Float32).sqrt();

        using var waveIn = new WasapiCapture();
        waveIn.WaveFormat = new WaveFormat(SampleRate, 16, 1);

        var playback = new BufferedWaveProvider(new WaveFormat(SampleRate, 16, 1))
        {
            DiscardOnBufferOverflow = true,
            BufferLength = SampleRate * 2
        };
        var output = new WaveOutEvent();
        output.Init(playback);
        output.Play();

        waveIn.DataAvailable += (s, e) =>
        {
            // Convert bytes to float
            int samplesCount = e.BytesRecorded / 2;
            float[] samples = new float[samplesCount];
            for (int i = 0; i < samplesCount; i++)
                samples[i] = BitConverter.ToInt16(e.Buffer, i * 2) / 32768f;

            var tensorWaveform = torch.tensor(samples, dtype: ScalarType.Float32);

            // STFT: [freq, time, 2]
            var stftTensor = torch.stft(tensorWaveform,
                n_fft: NFFT,
                hop_length: HopLength,
                win_length: WinLength,
                window: window,
                return_complex: false);

            // 推論每個 time frame
            var shape = stftTensor.shape; // [freq, time, 2]
            int freqBins = (int)shape[0];
            int frames = (int)shape[1];
            var inputData = stftTensor.data<float>().ToArray();

            var enhancedReal = new float[frames, freqBins];
            var enhancedImag = new float[frames, freqBins];

            for (int t = 0; t < frames; t++)
            {
                var frame = new DenseTensor<float>(new[] { 1, freqBins, 1, 2 });
                for (int f = 0; f < freqBins; f++)
                {
                    frame[0, f, 0, 0] = inputData[(f * frames + t) * 2];
                    frame[0, f, 0, 1] = inputData[(f * frames + t) * 2 + 1];
                }

                var results = session.Run(new[]
                {
                    NamedOnnxValue.CreateFromTensor("mix", frame),
                    NamedOnnxValue.CreateFromTensor("conv_cache", ToDenseTensor(conv_cache)),
                    NamedOnnxValue.CreateFromTensor("tra_cache", ToDenseTensor(tra_cache)),
                    NamedOnnxValue.CreateFromTensor("inter_cache", ToDenseTensor(inter_cache))
                });

                var outFrame = results[0].AsTensor<float>().ToArray();
                conv_cache = ToArray5D(results[1].AsTensor<float>(), conv_cache);
                tra_cache = ToArray5D(results[2].AsTensor<float>(), tra_cache);
                inter_cache = ToArray4D(results[3].AsTensor<float>(), inter_cache);

                for (int f = 0; f < freqBins; f++)
                {
                    enhancedReal[t, f] = outFrame[f * 2];
                    enhancedImag[t, f] = outFrame[f * 2 + 1];
                }
            }

            // Rebuild complex tensor
            var realTensor = torch.tensor(enhancedReal).transpose(0, 1);
            var imagTensor = torch.tensor(enhancedImag).transpose(0, 1);
            var complexTensor = torch.complex(realTensor, imagTensor);

            // ISTFT
            var istftOut = torch.istft(complexTensor,
                n_fft: NFFT,
                hop_length: HopLength,
                win_length: WinLength,
                window: window,
                length: samplesCount);

            // Write to playback buffer
            var outData = istftOut.data<float>().ToArray();
            byte[] pcmBytes = new byte[outData.Length * 2];
            for (int i = 0; i < outData.Length; i++)
            {
                short sVal = (short)Math.Clamp(outData[i] * 32767, short.MinValue, short.MaxValue);
                BitConverter.GetBytes(sVal).CopyTo(pcmBytes, i * 2);
            }
            playback.AddSamples(pcmBytes, 0, pcmBytes.Length);
        };

        waveIn.StartRecording();
        Console.WriteLine("🎤 Real-time GTCRN Denoising started... Press ENTER to stop.");
        Console.ReadLine();
        waveIn.StopRecording();
    }

    // ======= Helper =======
    static DenseTensor<float> ToDenseTensor(Array arr)
    {
        var shape = Enumerable.Range(0, arr.Rank).Select(arr.GetLength).ToArray();
        var flat = arr.Cast<float>().ToArray();
        return new DenseTensor<float>(flat, shape);
    }

    static float[,,,,] ToArray5D(Tensor<float> tensor, float[,,,,] template)
    {
        var flat = tensor.ToArray();
        var result = (float[,,,,])Array.CreateInstance(typeof(float), template.GetLength(0), template.GetLength(1),
                                                       template.GetLength(2), template.GetLength(3), template.GetLength(4));
        Buffer.BlockCopy(flat, 0, result, 0, flat.Length * sizeof(float));
        return result;
    }

    static float[,,,] ToArray4D(Tensor<float> tensor, float[,,,] template)
    {
        var flat = tensor.ToArray();
        var result = (float[,,,])Array.CreateInstance(typeof(float), template.GetLength(0), template.GetLength(1),
                                                      template.GetLength(2), template.GetLength(3));
        Buffer.BlockCopy(flat, 0, result, 0, flat.Length * sizeof(float));
        return result;
    }
}
