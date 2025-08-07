using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using NumSharp;
using Tensorflow;
using TorchSharp;
using static TorchSharp.torch;


class AudioDenoiser
{
    public static void Main(string[] args)
    {
        torch.random.manual_seed(42);
        torch.cuda.manual_seed(42);


        // ---- 讀取 WAV ----
        var reader = new AudioFileReader("./inp_16k.wav");
        var floatBuffer = new float[reader.Length / sizeof(float)];
        reader.Read(floatBuffer, 0, floatBuffer.Length);
        reader.Dispose();

        var waveform = torch.tensor(floatBuffer, dtype: ScalarType.Float32);

        // ---- STFT ----
        int n_fft = 512, hop_length = 256, win_length = 512;
        var window = torch.hann_window(n_fft, dtype: ScalarType.Float32).sqrt();
        var stftTensor = torch.stft(waveform,
                                    n_fft: n_fft,
                                    hop_length: hop_length,
                                    win_length: win_length,
                                    window: window,
                                    return_complex: false);
        Console.WriteLine($"STFT shape: {string.Join(",", stftTensor.shape)}"); // [Freq, Time, 2]
        var input = stftTensor.unsqueeze(0);  // shape: [1, 2, Freq, Time]

        // ---- ONNX Session ----
        var session = new InferenceSession("gtcrn_simple.onnx");

        var conv_cache = np.zeros(new NumSharp.Shape(2, 1, 16, 16, 33), dtype: np.float32);
        var tra_cache = np.zeros(new NumSharp.Shape(2, 3, 1, 1, 16), dtype: np.float32);
        var inter_cache = np.zeros(new NumSharp.Shape(2, 1, 33, 16), dtype: np.float32);

        var outputs = new List<float[,]>();

        //var inputTensor = input.permute(0, 2, 3, 1).contiguous(); // shape [1, Freq, Time, 2]
        var inputNp = input.cpu().data<float>().ToArray();
        var shape = input.shape;

        int frames = (int)shape[2];
        int freqBins = (int)shape[1];

        for (int i = 0; i < frames; i++)
        {
            // 取第 i frame 的複數 STFT（實部、虛部）
            var frame = new DenseTensor<float>(new[] { 1, freqBins, 1, 2 });
            for (int j = 0; j < freqBins; j++)
            {
                frame[0, j, 0, 0] = inputNp[(0 * freqBins * frames * 2) + (j * frames * 2) + (i * 2)];
                frame[0, j, 0, 1] = inputNp[(0 * freqBins * frames * 2) + (j * frames * 2) + (i * 2) + 1];
            }

            var result = session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("mix", frame),
                NamedOnnxValue.CreateFromTensor("conv_cache", ToDenseTensor(conv_cache)),
                NamedOnnxValue.CreateFromTensor("tra_cache", ToDenseTensor(tra_cache)),
                NamedOnnxValue.CreateFromTensor("inter_cache", ToDenseTensor(inter_cache))
            });

            var out_i = result[0].AsTensor<float>();
            conv_cache = ToNDArray(result[1].AsTensor<float>());
            tra_cache = ToNDArray(result[2].AsTensor<float>());
            inter_cache = ToNDArray(result[3].AsTensor<float>());

            var outFrame = new float[freqBins, 2];
            for (int j = 0; j < freqBins; j++)
            {
                outFrame[j, 0] = out_i[0, j, 0, 0];
                outFrame[j, 1] = out_i[0, j, 0, 1];
            }
            outputs.Add(outFrame);
        }

        // ---- 重建 Complex Tensor ----
        var outputReal = new float[outputs.Count, freqBins];
        var outputImag = new float[outputs.Count, freqBins];
        for (int t = 0; t < outputs.Count; t++)
        {
            for (int j = 0; j < freqBins; j++)
            {
                outputReal[t, j] = outputs[t][j, 0];
                outputImag[t, j] = outputs[t][j, 1];
            }
        }

        var realTensor = torch.tensor(outputReal).transpose(0, 1); // [freq, time]
        var imagTensor = torch.tensor(outputImag).transpose(0, 1);

        var complex = torch.complex(realTensor, imagTensor); // [freq, time]
        var istft_out = torch.istft(complex, n_fft: n_fft, hop_length: hop_length, win_length: win_length,
                                    window: window, length: waveform.shape[0]);

        // ---- 輸出 WAV ----
        using (var writer = new WaveFileWriter("./out_16k.wav", new WaveFormat(16000, 16, 1)))
        {
            foreach (var sample in istft_out.data<float>().ToArray())
            {
                writer.WriteSample(sample);
            }
        }

        Console.WriteLine("✅ 完成處理並儲存 out_16k.wav");

    }

    // --------- 輔助函數（NDArray 與 DenseTensor 互轉）---------
    static DenseTensor<float> ToDenseTensor(NDArray npArray)
    {
        var data = npArray.astype(np.float32).GetData<float>();
        return new DenseTensor<float>(data.ToArray(), npArray.shape);
    }
    static NDArray ToNDArray(Tensor<float> tensor)
    {
        return np.array(tensor.ToArray()).reshape(tensor.Dimensions.ToArray());
    }
}


