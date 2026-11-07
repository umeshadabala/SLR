import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

class AndroidNativeBridge {
  static const MethodChannel _channel = MethodChannel('com.example.exp07_slr/channel');

  static bool get isAndroid =>
      !kIsWeb && defaultTargetPlatform == TargetPlatform.android;

  static Future<Map<String, dynamic>?> getDeviceDiagnostics() async {
    if (!isAndroid) return null;
    try {
      final res = await _channel.invokeMethod<Map<dynamic, dynamic>>('getDeviceDiagnostics');
      return res?.cast<String, dynamic>();
    } catch (e) {
      debugPrint('getDeviceDiagnostics error: $e');
      return null;
    }
  }

  static Future<Map<String, dynamic>?> getMemoryMetrics() async {
    if (!isAndroid) return null;
    try {
      final res = await _channel.invokeMethod<Map<dynamic, dynamic>>('getMemoryMetrics');
      return res?.cast<String, dynamic>();
    } catch (e) {
      debugPrint('getMemoryMetrics error: $e');
      return null;
    }
  }

  static Future<Map<String, dynamic>?> getSafetyStatus() async {
    if (!isAndroid) return null;
    try {
      final raw = await _channel.invokeMethod<String>('getSafetyStatus');
      if (raw != null) {
        return jsonDecode(raw) as Map<String, dynamic>;
      }
      return null;
    } catch (e) {
      debugPrint('getSafetyStatus error: $e');
      return null;
    }
  }

  static Future<bool> configureSafetyPolicy({
    double maxProcessRssPercent = 75.0,
    double? maxProcessRssMb,
    double minAvailableSystemRamMb = 256.0,
    double memoryWarningPercent = 65.0,
    int samplingIntervalMs = 250,
  }) async {
    if (!isAndroid) return false;
    try {
      final ok = await _channel.invokeMethod<bool>('configureSafetyPolicy', {
        'maxProcessRssPercent': maxProcessRssPercent,
        'maxProcessRssMb': maxProcessRssMb,
        'minAvailableSystemRamMb': minAvailableSystemRamMb,
        'memoryWarningPercent': memoryWarningPercent,
        'samplingIntervalMs': samplingIntervalMs,
      });
      return ok ?? false;
    } catch (e) {
      debugPrint('configureSafetyPolicy error: $e');
      return false;
    }
  }

  static Future<bool> acknowledgeFuseReset() async {
    if (!isAndroid) return false;
    try {
      final ok = await _channel.invokeMethod<bool>('acknowledgeFuseReset');
      return ok ?? false;
    } catch (e) {
      debugPrint('acknowledgeFuseReset error: $e');
      return false;
    }
  }

  static Future<bool> triggerManualAbort(String reason) async {
    if (!isAndroid) return false;
    try {
      final ok = await _channel.invokeMethod<bool>('triggerManualAbort', {'reason': reason});
      return ok ?? false;
    } catch (e) {
      debugPrint('triggerManualAbort error: $e');
      return false;
    }
  }

  static Future<Map<String, dynamic>?> getStorageDirectory() async {
    if (!isAndroid) return null;
    try {
      final res = await _channel.invokeMethod<Map<dynamic, dynamic>>('getStorageDirectory');
      return res?.cast<String, dynamic>();
    } catch (e) {
      debugPrint('getStorageDirectory error: $e');
      return null;
    }
  }

  static Future<Map<String, dynamic>?> verifyCheckpoint() async {
    if (!isAndroid) return null;
    try {
      final res = await _channel.invokeMethod<Map<dynamic, dynamic>>('verifyCheckpoint');
      return res?.cast<String, dynamic>();
    } catch (e) {
      debugPrint('verifyCheckpoint error: $e');
      return null;
    }
  }

  static Future<Map<String, dynamic>?> loadModel({String? checkpointFile}) async {
    if (!isAndroid) return null;
    try {
      final res = await _channel.invokeMethod<Map<dynamic, dynamic>>('loadModel', {
        'checkpointFile': checkpointFile,
      });
      return res?.cast<String, dynamic>();
    } catch (e) {
      debugPrint('loadModel error: $e');
      rethrow;
    }
  }

  static Future<Map<String, dynamic>?> executeSingleLayer() async {
    if (!isAndroid) return null;
    try {
      final res = await _channel.invokeMethod<Map<dynamic, dynamic>>('executeSingleLayer');
      return res?.cast<String, dynamic>();
    } catch (e) {
      debugPrint('executeSingleLayer error: $e');
      rethrow;
    }
  }

  static Future<Map<String, dynamic>?> runSafeBenchmark() async {
    if (!isAndroid) return null;
    try {
      final raw = await _channel.invokeMethod<String>('runSafeBenchmark');
      if (raw != null) {
        return jsonDecode(raw) as Map<String, dynamic>;
      }
      return null;
    } catch (e) {
      debugPrint('runSafeBenchmark error: $e');
      rethrow;
    }
  }

  // --- Native SLR Generation & Streaming ---
  static bool _channelInitialized = false;
  static final StreamController<Map<String, dynamic>> _nativeTokenController =
      StreamController<Map<String, dynamic>>.broadcast();
  static final StreamController<Map<String, dynamic>> _nativeDoneController =
      StreamController<Map<String, dynamic>>.broadcast();
  static final StreamController<String> _nativeErrorController =
      StreamController<String>.broadcast();

  static void ensureInitialized() {
    if (_channelInitialized) return;
    _channelInitialized = true;
    _channel.setMethodCallHandler((call) async {
      switch (call.method) {
        case 'onNativeToken':
          if (call.arguments is Map) {
            _nativeTokenController.add(Map<String, dynamic>.from(call.arguments as Map));
          }
          break;
        case 'onNativeDone':
          if (call.arguments is Map) {
            _nativeDoneController.add(Map<String, dynamic>.from(call.arguments as Map));
          }
          break;
        case 'onNativeError':
          if (call.arguments is Map) {
            final args = Map<String, dynamic>.from(call.arguments as Map);
            _nativeErrorController.add(args['error']?.toString() ?? 'Native inference error');
          } else {
            _nativeErrorController.add(call.arguments?.toString() ?? 'Native inference error');
          }
          break;
      }
    });
  }

  static Stream<Map<String, dynamic>> get onNativeToken {
    ensureInitialized();
    return _nativeTokenController.stream;
  }

  static Stream<Map<String, dynamic>> get onNativeDone {
    ensureInitialized();
    return _nativeDoneController.stream;
  }

  static Stream<String> get onNativeError {
    ensureInitialized();
    return _nativeErrorController.stream;
  }

  static Future<bool> startNativeGeneration({
    required String prompt,
    List<Map<String, String>>? messages,
    int maxNewTokens = 32,
    double temperature = 0.0,
    double topP = 1.0,
  }) async {
    if (!isAndroid) return false;
    ensureInitialized();
    try {
      final res = await _channel.invokeMethod<bool>('startNativeGeneration', {
        'prompt': prompt,
        'messages': messages,
        'maxNewTokens': maxNewTokens,
        'temperature': temperature,
        'topP': topP,
      });
      return res ?? false;
    } catch (e) {
      debugPrint('startNativeGeneration error: $e');
      return false;
    }
  }

  static Future<bool> cancelNativeGeneration() async {
    if (!isAndroid) return false;
    try {
      final res = await _channel.invokeMethod<bool>('cancelNativeGeneration');
      return res ?? false;
    } catch (e) {
      debugPrint('cancelNativeGeneration error: $e');
      return false;
    }
  }

  static Future<List<Map<String, dynamic>>> listNativeRuns() async {
    if (!isAndroid) return [];
    try {
      final res = await _channel.invokeMethod<List<dynamic>>('listNativeRuns');
      if (res == null) return [];
      return res.map((item) => Map<String, dynamic>.from(item as Map)).toList();
    } catch (e) {
      debugPrint('listNativeRuns error: $e');
      return [];
    }
  }
}
