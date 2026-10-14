import 'dart:convert';
import 'package:http/http.dart' as http;
import '../../core/config/api_config.dart';
import '../models/backend_health.dart';
import '../models/benchmark_data.dart';
import '../models/experiment_data.dart';
import '../models/generation_job.dart';
import '../models/model_status.dart';
import '../models/telemetry_data.dart';

class SlrApiClient {
  final http.Client _client;

  SlrApiClient({http.Client? client}) : _client = client ?? http.Client();

  Uri _uri(String path) => Uri.parse('${ApiConfig.baseUrl}$path');

  Future<BackendHealth> checkHealth() async {
    try {
      final res = await _client.get(_uri('/health')).timeout(ApiConfig.timeout);
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        return BackendHealth.fromJson(data);
      }
      return BackendHealth.unreachable(error: 'HTTP ${res.statusCode}: ${res.body}');
    } catch (e) {
      return BackendHealth.unreachable(error: e.toString());
    }
  }

  Future<ModelStatus> getModelStatus() async {
    try {
      final res = await _client.get(_uri('/api/v1/model/status')).timeout(ApiConfig.timeout);
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        return ModelStatus.fromJson(data);
      } else {
        return ModelStatus.offline(error: 'HTTP ${res.statusCode}: ${res.body}');
      }
    } catch (e) {
      return ModelStatus.offline(error: e.toString());
    }
  }

  Future<bool> loadModel({String? checkpointDir, String? checkpointFile, String dtype = 'bfloat16'}) async {
    final body = jsonEncode({
      'checkpoint_dir': checkpointDir,
      'checkpoint_file': checkpointFile,
      'dtype': dtype,
    });
    final res = await _client.post(
      _uri('/api/v1/model/load'),
      headers: {'Content-Type': 'application/json'},
      body: body,
    ).timeout(const Duration(seconds: 30));
    return res.statusCode == 200;
  }

  Future<GenerationJob> startGeneration(String prompt, {int maxNewTokens = 8}) async {
    final body = jsonEncode({
      'prompt': prompt,
      'max_new_tokens': maxNewTokens,
    });
    final res = await _client.post(
      _uri('/api/v1/generation/start'),
      headers: {'Content-Type': 'application/json'},
      body: body,
    ).timeout(const Duration(seconds: 60));

    if (res.statusCode == 200) {
      return GenerationJob.fromJson(jsonDecode(res.body));
    } else {
      throw Exception('Generation failed (${res.statusCode}): ${res.body}');
    }
  }

  Stream<Map<String, dynamic>> streamGeneration({
    String? prompt,
    List<Map<String, dynamic>>? messages,
    int maxNewTokens = 64,
    double temperature = 0.0,
    double topP = 0.9,
    double repetitionPenalty = 1.0,
    bool doSample = false,
    String? jobId,
    double? clientRequestStartMs,
  }) async* {
    final request = http.Request('POST', _uri('/api/v1/generation/stream'));
    request.headers['Content-Type'] = 'application/json';
    final payload = <String, dynamic>{
      'max_new_tokens': maxNewTokens,
      'temperature': temperature,
      'top_p': topP,
      'repetition_penalty': repetitionPenalty,
      'do_sample': doSample,
    };
    if (prompt != null) payload['prompt'] = prompt;
    if (messages != null) payload['messages'] = messages;
    if (jobId != null) payload['job_id'] = jobId;
    if (clientRequestStartMs != null) payload['client_request_start_ms'] = clientRequestStartMs;
    request.body = jsonEncode(payload);

    final streamedResponse = await _client.send(request);
    if (streamedResponse.statusCode != 200) {
      final errorBody = await streamedResponse.stream.bytesToString();
      throw Exception('Streaming generation failed (${streamedResponse.statusCode}): $errorBody');
    }

    String buffer = '';
    await for (final chunk in streamedResponse.stream.transform(utf8.decoder)) {
      buffer += chunk;
      while (buffer.contains('\n\n')) {
        final index = buffer.indexOf('\n\n');
        final rawEvent = buffer.substring(0, index).trim();
        buffer = buffer.substring(index + 2);
        if (rawEvent.startsWith('data:')) {
          final jsonStr = rawEvent.substring(5).trim();
          if (jsonStr.isNotEmpty) {
            try {
              final parsed = jsonDecode(jsonStr) as Map<String, dynamic>;
              yield parsed;
            } catch (_) {}
          }
        }
      }
    }
  }

  Future<List<Map<String, dynamic>>> listRunReports({int limit = 50}) async {
    try {
      final res = await _client.get(_uri('/api/v1/results/runs?limit=$limit')).timeout(ApiConfig.timeout);
      if (res.statusCode == 200) {
        final list = jsonDecode(res.body) as List<dynamic>;
        return list.cast<Map<String, dynamic>>();
      }
    } catch (_) {}
    return [];
  }

  Future<Map<String, dynamic>?> getRunReport(String runId) async {
    try {
      final res = await _client.get(_uri('/api/v1/results/runs/$runId')).timeout(ApiConfig.timeout);
      if (res.statusCode == 200) {
        return jsonDecode(res.body) as Map<String, dynamic>;
      }
    } catch (_) {}
    return null;
  }

  Future<GenerationJob> getGenerationJob(String jobId) async {
    final res = await _client.get(_uri('/api/v1/generation/$jobId')).timeout(ApiConfig.timeout);
    if (res.statusCode == 200) {
      return GenerationJob.fromJson(jsonDecode(res.body));
    }
    throw Exception('Failed to get job: ${res.statusCode}');
  }

  Future<void> cancelGeneration(String jobId) async {
    await _client.post(_uri('/api/v1/generation/$jobId/cancel')).timeout(ApiConfig.timeout);
  }

  Future<TelemetryData> getTelemetry() async {
    try {
      final res = await _client.get(_uri('/api/v1/telemetry')).timeout(ApiConfig.timeout);
      if (res.statusCode == 200) {
        return TelemetryData.fromJson(jsonDecode(res.body));
      }
    } catch (_) {}
    return TelemetryData.unavailable();
  }

  Future<List<BenchmarkData>> getBenchmarks() async {
    try {
      final res = await _client.get(_uri('/api/v1/benchmarks')).timeout(ApiConfig.timeout);
      if (res.statusCode == 200) {
        final list = jsonDecode(res.body) as List<dynamic>;
        return list.map((e) => BenchmarkData.fromJson(e as Map<String, dynamic>)).toList();
      }
    } catch (_) {}
    return [];
  }

  Future<List<ExperimentData>> getExperiments() async {
    try {
      final res = await _client.get(_uri('/api/v1/experiments')).timeout(ApiConfig.timeout);
      if (res.statusCode == 200) {
        final list = jsonDecode(res.body) as List<dynamic>;
        return list.map((e) => ExperimentData.fromJson(e as Map<String, dynamic>)).toList();
      }
    } catch (_) {}
    return [];
  }

  Future<String> exportResultsJson() async {
    final res = await _client.get(_uri('/api/v1/results/export')).timeout(ApiConfig.timeout);
    if (res.statusCode == 200) {
      return res.body;
    }
    throw Exception('Failed to export results: ${res.statusCode}');
  }
}
