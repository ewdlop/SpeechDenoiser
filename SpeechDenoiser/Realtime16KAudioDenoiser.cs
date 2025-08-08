using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using NumSharp;
using TorchSharp;
using static TorchSharp.torch;
using System.Collections.Concurrent;

class Realtime16KAudioDenoiser
{
    const int SampleRate = 16000;
    const int n_fft = 512;
    const int hop_length = 256;
    const int win_length = 512;

    private static InferenceSession session;
    private static NDArray conv_cache;
    private static NDArray tra_cache;
    private static NDArray inter_cache;
    private static torch.Tensor window;

    private static BufferedWaveProvider outputBuffer;
    private static ConcurrentQueue<float> processQueue = new ConcurrentQueue<float>();

    public static void Main(string[] args)
    {
        // --- 初始化 ONNX Session ---
        session = new InferenceSession("gtcrn_simple.onnx");
        conv_cache = np.zeros(new Shape(2, 1, 16, 16, 33), np.float32);
        tra_cache = np.zeros(new Shape(2, 3, 1, 1, 16), np.float32);
        inter_cache = np.zeros(new Shape(2, 1, 33, 16), np.float32);
        window = hann_window(n_fft, dtype: ScalarType.Float32).sqrt();

        // --- 輸出播放緩衝區 ---
        outputBuffer = new BufferedWaveProvider(new WaveFormat(SampleRate, 16, 1))
        {
            DiscardOnBufferOverflow = true
        };

        var waveOut = new WaveOutEvent();
        waveOut.Init(outputBuffer);
        waveOut.Play();

        // --- 錄音設備 ---
        var waveIn = new WaveInEvent
        {
            WaveFormat = new WaveFormat(SampleRate, 16, 1),
            BufferMilliseconds = hop_length * 1000 / SampleRate // 每次 buffer 長度對應 hop_length
        };
        waveIn.DataAvailable += OnDataAvailable;
        waveIn.StartRecording();

        Console.WriteLine("🎤 即時降噪啟動中... 按 Enter 結束");
        Console.ReadLine();
        waveIn.StopRecording();
        waveOut.Stop();
    }

    private static void OnDataAvailable(object sender, WaveInEventArgs e)
    {
        // 把 byte[] 轉 float[]
        var floatBuffer = new float[e.BytesRecorded / 2];
        for (int i = 0; i < floatBuffer.Length; i++)
            floatBuffer[i] = BitConverter.ToInt16(e.Buffer, i * 2) / 32768f;

        foreach (var sample in floatBuffer)
            processQueue.Enqueue(sample);

        // 處理足夠的 frame
        while (processQueue.Count >= hop_length)
        {
            var frame = new float[hop_length];
            for (int i = 0; i < hop_length; i++)
                processQueue.TryDequeue(out frame[i]);

            var denoised = ProcessFrame(frame);
            // 寫回播放 buffer
            var outBytes = new byte[denoised.Length * 2];
            for (int i = 0; i < denoised.Length; i++)
            {
                short pcm = (short)(Math.Max(-1, Math.Min(1, denoised[i])) * 32767);
                BitConverter.GetBytes(pcm).CopyTo(outBytes, i * 2);
            }
            outputBuffer.AddSamples(outBytes, 0, outBytes.Length);
        }
    }

    private static float[] ProcessFrame(float[] frame)
    {
        // STFT（單 frame 需要保留過去 win_length 的資料，這裡簡化成單 frame 處理）
        var waveform = tensor(frame, dtype: ScalarType.Float32);
        var stftTensor = stft(waveform,
                               n_fft: n_fft,
                               hop_length: hop_length,
                               win_length: win_length,
                               window: window,
                               return_complex: false);
        var freqBins = (int)stftTensor.shape[0];

        var frameTensor = new DenseTensor<float>(new[] { 1, freqBins, 1, 2 });
        var data = stftTensor.data<float>().ToArray();
        for (int j = 0; j < freqBins; j++)
        {
            frameTensor[0, j, 0, 0] = data[j * 2];
            frameTensor[0, j, 0, 1] = data[j * 2 + 1];
        }

        var result = session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("mix", frameTensor),
            NamedOnnxValue.CreateFromTensor("conv_cache", ToDenseTensor(conv_cache)),
            NamedOnnxValue.CreateFromTensor("tra_cache", ToDenseTensor(tra_cache)),
            NamedOnnxValue.CreateFromTensor("inter_cache", ToDenseTensor(inter_cache))
        });

        var out_i = result[0].AsTensor<float>();
        conv_cache = ToNDArray(result[1].AsTensor<float>());
        tra_cache = ToNDArray(result[2].AsTensor<float>());
        inter_cache = ToNDArray(result[3].AsTensor<float>());

        var real = new float[freqBins];
        var imag = new float[freqBins];
        for (int j = 0; j < freqBins; j++)
        {
            real[j] = out_i[0, j, 0, 0];
            imag[j] = out_i[0, j, 0, 1];
        }

        var complexOut = complex(tensor(real), tensor(imag));
        var istft_out = istft(complexOut,
                              n_fft: n_fft,
                              hop_length: hop_length,
                              win_length: win_length,
                              window: window,
                              length: frame.Length);

        return istft_out.data<float>().ToArray();
    }

    private static DenseTensor<float> ToDenseTensor(NDArray npArray)
    {
        var data = npArray.astype(np.float32).GetData<float>();
        return new DenseTensor<float>(data.ToArray(), npArray.shape);
    }

    private static NDArray ToNDArray(Tensor<float> tensor)
    {
        return np.array(tensor.ToArray()).reshape(tensor.Dimensions.ToArray());
    }
}
