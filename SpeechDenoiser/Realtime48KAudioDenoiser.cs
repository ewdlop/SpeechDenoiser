using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;

class Realtime48KAudioDenoiser : IDisposable
{
    // ======== 依模型定義調整 ========
    const int SampleRate = 48000;
    const int Channels = 1;
    const int HopSize = 480;   // 10 ms @ 48k
    const int FftSize = 960;   // 用於推算延遲：960 - 480 = 480
    // 若你確定模型的 state 長度固定如下，保留；如果想動態偵測，見下面註解。
    const int StateLength = 45304;

    private readonly InferenceSession _session;
    private readonly WaveInEvent _waveIn;
    private readonly WaveOutEvent _waveOut;
    private readonly BufferedWaveProvider _playBuffer;
    private readonly BlockingCollection<float[]> _inQueue = new(256);
    private readonly Thread _worker;
    private readonly CancellationTokenSource _cts = new();
    private float[] _state = new float[StateLength];
    private readonly float[] _attenLimDb = new float[1] { 0f };

    // 累積器：把 NAudio 給的 bytes 轉 float，累到 480 即送推論
    private readonly List<float> _accumulator = new(HopSize * 4);

    public Realtime48KAudioDenoiser(string onnxPath)
    {
        var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            IntraOpNumThreads = 1,
            InterOpNumThreads = 1,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
        };
        _session = new InferenceSession(onnxPath, so);

         var meta = _session.InputMetadata["states"];
        int dynLen = (int)(meta.Dimensions.Last() <= 0 ? StateLength : meta.Dimensions.Last());
        _state = new float[dynLen];

        // 設定錄音端（10ms buffer）
        _waveIn = new WaveInEvent
        {
            WaveFormat = new WaveFormat(SampleRate, 16, Channels),
            BufferMilliseconds = 10,
            NumberOfBuffers = 4
        };
        _waveIn.DataAvailable += OnData;

        // 設定播放端
        _playBuffer = new BufferedWaveProvider(new WaveFormat(SampleRate, 16, Channels))
        {
            BufferLength = HopSize * 2 * 100, // 100 個 frame 的空間
            DiscardOnBufferOverflow = true
        };
        _waveOut = new WaveOutEvent();
        _waveOut.Init(_playBuffer);

        // 推論/播放工作執行緒
        _worker = new Thread(WorkerLoop) { IsBackground = true, Name = "AEC-Denoise-Worker" };
    }

    public void Start()
    {
        _waveOut.Play();

        // 啟動時先推入延遲補償（480 zeros）
        WritePcmToPlayBuffer(new float[FftSize - HopSize]); // 960-480=480

        _worker.Start();
        _waveIn.StartRecording();
        Console.WriteLine("Recording+Denoising started. Press ENTER to stop.");
    }

    public void Stop()
    {
        _waveIn?.StopRecording();
        _cts.Cancel();
        _inQueue.CompleteAdding();
        _worker.Join();
        _waveOut?.Stop();
    }

    public void Dispose()
    {
        try { Stop(); } catch { /* ignore */ }
        _waveIn?.Dispose();
        _waveOut?.Dispose();
        _session?.Dispose();
        _cts.Dispose();
    }

    private void OnData(object? sender, WaveInEventArgs e)
    {
        // 16-bit PCM -> float (-1..1)
        int samples = e.BytesRecorded / 2;
        for (int i = 0; i < samples; i++)
        {
            short s = BitConverter.ToInt16(e.Buffer, i * 2);
            _accumulator.Add(s / 32768f);
            if (_accumulator.Count >= HopSize)
            {
                var frame = _accumulator.Take(HopSize).ToArray();
                _accumulator.RemoveRange(0, HopSize);

                // 推給 worker；若塞滿，丟棄一個最舊的避免卡住
                if (!_inQueue.TryAdd(frame))
                {
                    _inQueue.TryTake(out _);
                    _inQueue.TryAdd(frame);
                }
            }
        }
    }

    private void WorkerLoop()
    {
        try
        {
            foreach (var frame in _inQueue.GetConsumingEnumerable(_cts.Token))
            {
                var enhanced = Infer(frame, ref _state, _attenLimDb);
                WritePcmToPlayBuffer(enhanced);
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Worker error: {ex}");
        }
    }

    private float[] Infer(float[] frame480, ref float[] state, float[] attenLimDb)
    {
        // ONNX 輸入張量
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_frame",
                new DenseTensor<float>(frame480, new int[]{ frame480.Length })),
            NamedOnnxValue.CreateFromTensor("states",
                new DenseTensor<float>(state, new int[]{ state.Length })),
            NamedOnnxValue.CreateFromTensor("atten_lim_db",
                new DenseTensor<float>(attenLimDb, new int[]{ 1 }))
        };

        using var results = _session.Run(inputs);
        var outList = results.ToList();

        // 假設輸出[0]為強健後 frame，輸出[1]為下一輪 state
        var enhanced = outList[0].AsTensor<float>().ToArray();
        state = outList[1].AsTensor<float>().ToArray();

        return enhanced;
    }

    private void WritePcmToPlayBuffer(float[] mono)
    {
        // float(-1..1) -> 16-bit PCM 小端
        var bytes = new byte[mono.Length * 2];
        for (int i = 0, j = 0; i < mono.Length; i++, j += 2)
        {
            float clamped = MathF.Max(-1f, MathF.Min(1f, mono[i]));
            short s = (short)(clamped * short.MaxValue);
            bytes[j] = (byte)(s & 0xFF);
            bytes[j + 1] = (byte)((s >> 8) & 0xFF);
        }
        _playBuffer.AddSamples(bytes, 0, bytes.Length);
    }

    public static void Main(string[] args)
    {
        string modelPath = args.Length > 0 ? args[0] : "./denoiser_model.onnx";
        using var app = new Realtime48KAudioDenoiser(modelPath);
        app.Start();
        Console.ReadLine();
        app.Stop();
        Console.WriteLine("Stopped.");
    }
}
