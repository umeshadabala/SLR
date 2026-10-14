import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import '../../core/config/api_config.dart';
import '../../core/native/android_bridge.dart';
import '../api/slr_api_client.dart';
import '../models/backend_health.dart';
import '../models/benchmark_data.dart';
import '../models/chat_message.dart';
import '../models/experiment_data.dart';
import '../models/generation_job.dart';
import '../models/model_status.dart';
import '../models/system_status.dart';
import '../models/telemetry_data.dart';

class SlrRepository extends ChangeNotifier {
  final SlrApiClient _api;

  BackendHealth _backendHealth = BackendHealth.unreachable();
  ModelStatus _modelStatus = ModelStatus.offline();
  TelemetryData _telemetry = TelemetryData.unavailable();
  List<BenchmarkData> _benchmarks = [];
  List<ExperimentData> _experiments = [];
  GenerationJob? _currentJob;
  bool _isGenerating = false;
  String? _lastError;
  Timer? _telemetryTimer;

  // Conversational Chatbot State
  final List<ChatMessageItem> _chatMessages = [];
  StreamSubscription? _streamSubscription;
  String? _activeStreamingJobId;

  // Configurable Generation Settings
  int _maxNewTokens = 64;
  double _temperature = 0.7;
  double _topP = 0.9;
  bool _greedyDecoding = true;
  double _repetitionPenalty = 1.05;

  // Android On-Device state
  Map<String, dynamic>? _androidDiagnostics;
  Map<String, dynamic>? _safetyStatus;
  Map<String, dynamic>? _lastBenchmarkReport;
  bool _isRunningBenchmark = false;

  SlrRepository({SlrApiClient? api}) : _api = api ?? SlrApiClient();

  BackendHealth get backendHealth => _backendHealth;
  ModelStatus get modelStatus => _modelStatus;
  TelemetryData get telemetry => _telemetry;
  List<BenchmarkData> get benchmarks => _benchmarks;
  List<ExperimentData> get experiments => _experiments;
  GenerationJob? get currentJob => _currentJob;
  bool get isGenerating => _isGenerating;
  bool get isRunningBenchmark => _isRunningBenchmark;
  String? get lastError => _lastError;
  bool get isConnected => _backendHealth.isReachable;

  SlrSystemStatus get systemStatus {
    if (_isGenerating) return SlrSystemStatus.inferenceRunning;
    if (!_backendHealth.isReachable) return SlrSystemStatus.backendUnreachable;
    if (_lastError != null && _lastError!.isNotEmpty && !_modelStatus.isLoaded) {
      return SlrSystemStatus.inferenceFailed;
    }
    if (_backendHealth.isReachable && !_backendHealth.modelLoaded) {
      if (_backendHealth.modelState == 'checkpoint_missing' || _modelStatus.state == 'checkpoint_missing') {
        return SlrSystemStatus.modelMissing;
      }
      if (_backendHealth.modelState == 'loading' || _modelStatus.state == 'loading') {
        return SlrSystemStatus.modelLoading;
      }
      if (_backendHealth.modelState == 'error' || _modelStatus.state == 'error') {
        return SlrSystemStatus.modelVerifiedIncompatible;
      }
      return SlrSystemStatus.backendReachable;
    }
    if (_backendHealth.modelLoaded || _modelStatus.isLoaded) {
      return SlrSystemStatus.modelReady;
    }
    return SlrSystemStatus.backendReachable;
  }

  // Chatbot getters
  List<ChatMessageItem> get chatMessages => List.unmodifiable(_chatMessages);
  int get maxNewTokens => _maxNewTokens;
  double get temperature => _temperature;
  double get topP => _topP;
  bool get greedyDecoding => _greedyDecoding;
  double get repetitionPenalty => _repetitionPenalty;

  Map<String, dynamic>? get androidDiagnostics => _androidDiagnostics;
  Map<String, dynamic>? get safetyStatus => _safetyStatus;
  Map<String, dynamic>? get lastBenchmarkReport => _lastBenchmarkReport;
  bool get isAndroidRuntime => AndroidNativeBridge.isAndroid;

  void setMaxNewTokens(int val) {
    _maxNewTokens = val;
    notifyListeners();
  }

  void setTemperature(double val) {
    _temperature = val;
    notifyListeners();
  }

  void setTopP(double val) {
    _topP = val;
    notifyListeners();
  }

  void setGreedyDecoding(bool val) {
    _greedyDecoding = val;
    notifyListeners();
  }

  void setRepetitionPenalty(double val) {
    _repetitionPenalty = val;
    notifyListeners();
  }

  void clearChat() {
    _streamSubscription?.cancel();
    _streamSubscription = null;
    _isGenerating = false;
    _chatMessages.clear();
    _currentJob = null;
    _lastError = null;
    notifyListeners();
  }

  void startTelemetryPolling() {
    _telemetryTimer?.cancel();
    _telemetryTimer = Timer.periodic(ApiConfig.telemetryPollInterval, (_) async {
      await refreshTelemetry();
    });
  }

  void stopTelemetryPolling() {
    _telemetryTimer?.cancel();
    _telemetryTimer = null;
  }

  Future<void> retryConnection() async {
    _lastError = null;
    notifyListeners();
    await refreshAll();
  }

  Future<void> refreshAll() async {
    await refreshModelStatus();
    await refreshTelemetry();
    if (_backendHealth.isReachable) {
      await refreshBenchmarks();
      await refreshExperiments();
    }
  }

  Future<void> refreshModelStatus() async {
    // 1. Probe backend health unconditionally to verify actual reachability
    final health = await _api.checkHealth();
    _backendHealth = health;

    if (health.isReachable) {
      // Backend is reachable - retrieve exact model metadata
      _modelStatus = await _api.getModelStatus();
      if (AndroidNativeBridge.isAndroid) {
        _androidDiagnostics = await AndroidNativeBridge.getDeviceDiagnostics();
      }
      notifyListeners();
    } else {
      // Backend is unreachable
      if (AndroidNativeBridge.isAndroid) {
        _androidDiagnostics = await AndroidNativeBridge.getDeviceDiagnostics();
        final chk = await AndroidNativeBridge.verifyCheckpoint();
        final localFileExists = chk?['exists'] == true;

        _modelStatus = ModelStatus(
          modelId: 'Qwen/Qwen2.5-1.5B-Instruct',
          targetRevision: '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
          state: 'backend_unreachable',
          isLoaded: false,
          isSimulated: false,
          engineType: 'SLR Inference Engine (Host Disconnected)',
          dtype: 'bfloat16',
          totalParameters: 1543714304,
          activeParameters: 0,
          activeLayers: 0,
          totalLayers: 28,
          layerSizeMb: 89.26,
          errorMessage: 'Cannot connect to SLR backend at ${ApiConfig.baseUrl}: ${health.error ?? "Connection refused"}',
          checkpointGuidance: localFileExists
              ? 'Local weights exist on phone storage. To connect to host backend over USB, run "adb reverse tcp:8000 tcp:8000" on your Mac, or configure host IP in Settings.'
              : 'SLR backend unreachable at ${ApiConfig.baseUrl}. Set up "adb reverse tcp:8000 tcp:8000" or configure Mac Wi-Fi IP in Settings.',
        );
      } else {
        _modelStatus = ModelStatus.offline(
          error: 'Cannot connect to SLR backend at ${ApiConfig.baseUrl}: ${health.error ?? "Connection refused"}',
        );
      }
      notifyListeners();
    }
  }

  Future<void> refreshTelemetry() async {
    if (_backendHealth.isReachable) {
      _telemetry = await _api.getTelemetry();
      notifyListeners();
    } else if (AndroidNativeBridge.isAndroid) {
      final mem = await AndroidNativeBridge.getMemoryMetrics();
      _safetyStatus = await AndroidNativeBridge.getSafetyStatus();

      if (mem != null) {
        final rssMb = (mem['processRssMb'] as num?)?.toDouble() ?? 0.0;
        final rssGb = rssMb / 1024.0;
        final activeIdx = _safetyStatus?['current_active_layer_idx'] as int? ?? -1;
        final isLayerActive = activeIdx >= 0;
        final kvLen = _safetyStatus?['current_kv_cache_length'] as int? ?? 0;

        _telemetry = TelemetryData(
          isEngineActive: isLayerActive,
          currentLayerIdx: isLayerActive ? activeIdx : null,
          totalLayers: 28,
          activeTransformerLayers: isLayerActive ? 1 : 0,
          activeTransformerParams: isLayerActive ? 46797824 : 0,
          processRssGb: rssGb,
          processRssMb: rssMb,
          kvCacheSequenceLength: kvLen,
          engineMode: 'android_slr_native',
          isSimulated: false,
          timestamp: DateTime.now().millisecondsSinceEpoch / 1000.0,
        );
      }
      notifyListeners();
    } else {
      _telemetry = TelemetryData.unavailable();
      notifyListeners();
    }
  }

  Future<void> refreshBenchmarks() async {
    if (_backendHealth.isReachable) {
      _benchmarks = await _api.getBenchmarks();
      notifyListeners();
    }
  }

  Future<void> refreshExperiments() async {
    if (_backendHealth.isReachable) {
      _experiments = await _api.getExperiments();
      notifyListeners();
    }
  }

  Future<Map<String, dynamic>?> runAndroidSafeBenchmark() async {
    if (!AndroidNativeBridge.isAndroid) return null;
    _isRunningBenchmark = true;
    _lastError = null;
    notifyListeners();

    try {
      final report = await AndroidNativeBridge.runSafeBenchmark();
      _lastBenchmarkReport = report;

      if (report != null && report.containsKey('tests')) {
        final tests = report['tests'] as List<dynamic>? ?? [];
        final summary = report['summary'] as Map<String, dynamic>? ?? {};

        // Accurate structural suite recording: tokensPerSecond is null because it is not a generation benchmark
        _benchmarks.insert(
          0,
          BenchmarkData(
            benchmarkId: 'BM-STRUCTURAL-ANDROID',
            experimentId: 'EXP-07-ON-DEVICE',
            title: '18-Test Automated Android Safe Benchmark (Structural Suite)',
            timestamp: DateTime.now().toIso8601String(),
            model: 'Qwen2.5-1.5B-Instruct',
            precision: 'BF16',
            device: '${report['device_diagnostics']?['manufacturer']} ${report['device_diagnostics']?['model']}',
            osPlatform: 'Android ${report['device_diagnostics']?['android_version']}',
            promptLength: 0,
            outputLength: 0,
            tokensPerSecond: null, // Structural suite: null to accurately reflect non-generation testing
            totalGenerationSec: ((summary['duration_ms'] as num?)?.toDouble() ?? 0.0) / 1000.0,
            peakRssGb: ((_telemetry.processRssGb > 0 ? _telemetry.processRssGb : 0.0)),
            activeTransformerParamsRatio: '1/28 (46.8M / 1.31B)',
            validationStatus: summary['final_status']?.toString() ?? 'COMPLETED',
            isSimulated: false,
            notes: tests.map((t) => '${t['test_id']}: ${t['test_name']} -> ${t['status']} (${t['details']})').cast<String>().toList(),
          ),
        );
      }

      await refreshTelemetry();
      return report;
    } catch (e) {
      _lastError = e.toString();
      return null;
    } finally {
      _isRunningBenchmark = false;
      notifyListeners();
    }
  }

  Future<Map<String, dynamic>?> executeSingleLayerOnDevice() async {
    if (!AndroidNativeBridge.isAndroid) return null;
    _lastError = null;
    notifyListeners();
    try {
      final res = await AndroidNativeBridge.executeSingleLayer();
      await refreshTelemetry();
      return res;
    } catch (e) {
      _lastError = e.toString();
      notifyListeners();
      return null;
    }
  }

  Future<bool> acknowledgeFuseReset() async {
    final ok = await AndroidNativeBridge.acknowledgeFuseReset();
    await refreshTelemetry();
    return ok;
  }

  Future<bool> triggerManualAbort(String reason) async {
    final ok = await AndroidNativeBridge.triggerManualAbort(reason);
    await refreshTelemetry();
    return ok;
  }

  Future<bool> loadModel({String? checkpointDir, String? checkpointFile}) async {
    _lastError = null;
    notifyListeners();
    try {
      if (_backendHealth.isReachable) {
        final ok = await _api.loadModel(
          checkpointDir: checkpointDir,
          checkpointFile: checkpointFile,
        );
        await refreshModelStatus();
        return ok;
      } else if (AndroidNativeBridge.isAndroid) {
        final res = await AndroidNativeBridge.loadModel(checkpointFile: checkpointFile);
        final success = res?['success'] == true;
        await refreshModelStatus();
        return success;
      } else {
        _lastError = 'Backend is unreachable at ${ApiConfig.baseUrl}';
        notifyListeners();
        return false;
      }
    } catch (e) {
      _lastError = e.toString();
      notifyListeners();
      return false;
    }
  }

  Future<void> sendChatMessage(String prompt) async {
    final trimmed = prompt.trim();
    if (trimmed.isEmpty || _isGenerating) return;

    _lastError = null;

    // Pre-flight check: ensure backend is reachable before inserting generation messages
    if (!_backendHealth.isReachable) {
      _lastError =
          'Cannot send prompt: SLR backend is unreachable at ${ApiConfig.baseUrl}. '
          '${AndroidNativeBridge.isAndroid ? "Run 'adb reverse tcp:8000 tcp:8000' over USB or configure host IP in Settings." : "Please start the Python backend server."}';
      notifyListeners();
      return;
    }

    if (!_modelStatus.isLoaded) {
      _lastError = 'Cannot send prompt: Model is not loaded (current state: "${_modelStatus.state}").';
      notifyListeners();
      return;
    }

    final userMsg = ChatMessageItem(
      id: 'usr_${DateTime.now().millisecondsSinceEpoch}',
      role: 'user',
      content: trimmed,
      timestamp: DateTime.now(),
      status: ChatMessageStatus.completed,
    );
    _chatMessages.add(userMsg);

    final assistantMsg = ChatMessageItem(
      id: 'asst_${DateTime.now().millisecondsSinceEpoch}',
      role: 'assistant',
      content: '',
      timestamp: DateTime.now(),
      status: ChatMessageStatus.generating,
    );
    _chatMessages.add(assistantMsg);
    _isGenerating = true;
    notifyListeners();

    try {
      final effectiveMessages = _chatMessages
          .where((m) => m != assistantMsg && !m.hasError)
          .map((m) => m.toApiJson())
          .toList();

      final jobId = 'chat_${DateTime.now().millisecondsSinceEpoch}';
      _activeStreamingJobId = jobId;

      final stream = _api.streamGeneration(
        messages: effectiveMessages,
        maxNewTokens: _maxNewTokens,
        temperature: _greedyDecoding ? 0.0 : _temperature,
        topP: _topP,
        repetitionPenalty: _repetitionPenalty,
        doSample: !_greedyDecoding && _temperature > 0.0,
        jobId: jobId,
      );

      _streamSubscription = stream.listen(
        (event) {
          final type = event['type']?.toString();
          if (type == 'token') {
            final text = event['text']?.toString() ?? '';
            assistantMsg.content += text;
            notifyListeners();
          } else if (type == 'done') {
            assistantMsg.status = ChatMessageStatus.completed;
            final resp = event['response'] as Map<String, dynamic>? ?? {};
            assistantMsg.timeToFirstTokenSec = (resp['time_to_first_token_sec'] as num?)?.toDouble();
            assistantMsg.decodeThroughput = (resp['decode_throughput'] as num?)?.toDouble() ??
                (resp['tokens_per_second'] as num?)?.toDouble();
            assistantMsg.totalGenerationSec = (resp['total_generation_sec'] as num?)?.toDouble();
            assistantMsg.promptTokens = resp['prompt_tokens'] as int?;
            assistantMsg.generatedTokens = resp['generated_tokens'] as int?;
            assistantMsg.peakRssGb = (resp['peak_process_rss_gb'] as num?)?.toDouble();
            assistantMsg.stopReason = resp['stop_reason']?.toString() ?? 'eos';

            _currentJob = GenerationJob.fromJson(resp);
            notifyListeners();
          } else if (type == 'error') {
            assistantMsg.status = ChatMessageStatus.error;
            assistantMsg.error = event['error']?.toString() ?? 'Generation error occurred.';
            _lastError = assistantMsg.error;
            notifyListeners();
          }
        },
        onError: (err) {
          assistantMsg.status = ChatMessageStatus.error;
          final errorStr = err.toString();
          if (errorStr.contains('Connection refused') || errorStr.contains('SocketException')) {
            assistantMsg.error =
                'Connection refused to SLR backend at ${ApiConfig.baseUrl}. '
                '${AndroidNativeBridge.isAndroid ? "Verify 'adb reverse tcp:8000 tcp:8000' is active." : "Verify server is running."}';
          } else {
            assistantMsg.error = errorStr;
          }
          _lastError = assistantMsg.error;
          _isGenerating = false;
          notifyListeners();
        },
        onDone: () {
          if (assistantMsg.status == ChatMessageStatus.generating) {
            assistantMsg.status = ChatMessageStatus.completed;
          }
          _isGenerating = false;
          refreshTelemetry();
          notifyListeners();
        },
      );
    } catch (e) {
      assistantMsg.status = ChatMessageStatus.error;
      assistantMsg.error = 'Inference not executed: ${e.toString()}';
      _lastError = assistantMsg.error;
      _isGenerating = false;
      notifyListeners();
    }
  }

  Future<void> retryFailedMessage(ChatMessageItem failedAsstMsg) async {
    final asstIdx = _chatMessages.indexOf(failedAsstMsg);
    if (asstIdx <= 0) return;

    final userMsg = _chatMessages[asstIdx - 1];
    final promptText = userMsg.content;

    // Remove failed assistant message and preceding user message to prevent duplicates
    _chatMessages.removeAt(asstIdx);
    _chatMessages.removeAt(asstIdx - 1);
    notifyListeners();

    // Re-send cleanly
    await sendChatMessage(promptText);
  }

  Future<void> cancelGeneration() async {
    if (_streamSubscription != null) {
      await _streamSubscription?.cancel();
      _streamSubscription = null;
    }
    if (_activeStreamingJobId != null) {
      try {
        await _api.cancelGeneration(_activeStreamingJobId!);
      } catch (_) {}
    }
    if (_currentJob != null && _isGenerating) {
      try {
        await _api.cancelGeneration(_currentJob!.jobId);
      } catch (_) {}
    }
    for (final m in _chatMessages) {
      if (m.isGenerating) {
        m.status = ChatMessageStatus.cancelled;
        m.stopReason = 'cancelled';
      }
    }
    _isGenerating = false;
    notifyListeners();
  }

  Future<String> exportResultsJson() async {
    if (AndroidNativeBridge.isAndroid && _lastBenchmarkReport != null) {
      return jsonEncode(_lastBenchmarkReport);
    }
    return await _api.exportResultsJson();
  }

  @override
  void dispose() {
    _streamSubscription?.cancel();
    stopTelemetryPolling();
    super.dispose();
  }
}
