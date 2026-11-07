import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/config/api_config.dart';
import '../../core/theme/app_colors.dart';
import '../../data/repositories/slr_repository.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _checkpointController = TextEditingController(
    text: 'checkpoints/Qwen2.5-1.5B-Instruct/model.safetensors',
  );
  final _urlController = TextEditingController(text: ApiConfig.baseUrl);

  @override
  void dispose() {
    _checkpointController.dispose();
    _urlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final repo = context.watch<SlrRepository>();
    final status = repo.modelStatus;

    return Scaffold(
      appBar: AppBar(
        title: const Text('ENGINE CONFIGURATION & DIAGNOSTICS'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Execution Mode Switcher
            Container(
              padding: const EdgeInsets.all(18),
              margin: const EdgeInsets.only(bottom: 20),
              decoration: BoxDecoration(
                color: AppColors.surfaceDark,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                  color: repo.isAndroidNativeMode ? AppColors.success : AppColors.primary,
                  width: 1.2,
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        repo.isAndroidNativeMode ? Icons.phone_android : Icons.computer,
                        color: repo.isAndroidNativeMode ? AppColors.success : AppColors.primary,
                        size: 20,
                      ),
                      const SizedBox(width: 8),
                      const Text(
                        'EXECUTION MODE',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 0.8,
                          color: Colors.white,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  RadioListTile<ExecutionMode>(
                    value: ExecutionMode.androidNative,
                    groupValue: repo.executionMode,
                    activeColor: AppColors.success,
                    title: const Text(
                      'ANDROID-NATIVE INFERENCE (Standalone On-Device)',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                    ),
                    subtitle: const Text(
                      '100% offline Qwen2.5-1.5B SLR engine running on device processor. No Mac backend or network required.',
                      style: TextStyle(fontSize: 11, color: AppColors.neutral),
                    ),
                    onChanged: (mode) {
                      if (mode != null) repo.setExecutionMode(mode);
                    },
                  ),
                  const Divider(height: 1),
                  RadioListTile<ExecutionMode>(
                    value: ExecutionMode.macHosted,
                    groupValue: repo.executionMode,
                    activeColor: AppColors.primary,
                    title: const Text(
                      'MAC-HOSTED SLR INFERENCE (Development Host)',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                    ),
                    subtitle: const Text(
                      'Stream tokens from Python FastAPI / PyTorch backend running on host Mac via USB adb reverse or Wi-Fi.',
                      style: TextStyle(fontSize: 11, color: AppColors.neutral),
                    ),
                    onChanged: (mode) {
                      if (mode != null) repo.setExecutionMode(mode);
                    },
                  ),
                ],
              ),
            ),

            // Checkpoint Configuration
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: AppColors.surfaceDark,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: AppColors.borderDark),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'CHECKPOINT PATH',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.8,
                      color: AppColors.neutral,
                    ),
                  ),
                  const SizedBox(height: 8),
                  TextField(
                    controller: _checkpointController,
                    style: const TextStyle(fontSize: 13, fontFamily: 'monospace'),
                    decoration: const InputDecoration(
                      hintText: 'Enter absolute or relative path to model.safetensors',
                      border: OutlineInputBorder(),
                      contentPadding: EdgeInsets.all(12),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    alignment: WrapAlignment.spaceBetween,
                    crossAxisAlignment: WrapCrossAlignment.center,
                    children: [
                      Text(
                        'Current Status: ${status.state.toUpperCase()}',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: status.isLoaded ? AppColors.success : AppColors.warning,
                        ),
                      ),
                      ElevatedButton.icon(
                        onPressed: () async {
                          final ok = await repo.loadModel(
                            checkpointFile: _checkpointController.text.trim(),
                          );
                          if (context.mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text(
                                  ok ? 'Model loaded successfully!' : 'Failed to load checkpoint.',
                                ),
                                backgroundColor: ok ? AppColors.success : AppColors.error,
                              ),
                            );
                          }
                        },
                        icon: const Icon(Icons.cloud_upload, size: 16),
                        label: const Text('LOAD MODEL'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppColors.primary,
                          foregroundColor: Colors.white,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // Backend Connection
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: AppColors.surfaceDark,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: AppColors.borderDark),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'BACKEND API ENDPOINT',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.8,
                      color: AppColors.neutral,
                    ),
                  ),
                  const SizedBox(height: 8),
                  TextField(
                    controller: _urlController,
                    style: const TextStyle(fontSize: 13, fontFamily: 'monospace'),
                    decoration: const InputDecoration(
                      hintText: 'http://127.0.0.1:8000',
                      border: OutlineInputBorder(),
                      contentPadding: EdgeInsets.all(12),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 6,
                    children: [
                      ActionChip(
                        label: const Text('127.0.0.1:8000 (USB)', style: TextStyle(fontSize: 11)),
                        onPressed: () {
                          _urlController.text = 'http://127.0.0.1:8000';
                          ApiConfig.setBaseUrl(_urlController.text);
                          repo.refreshAll();
                        },
                      ),
                      ActionChip(
                        label: const Text('192.168.1.13:8000 (Wi-Fi)', style: TextStyle(fontSize: 11)),
                        onPressed: () {
                          _urlController.text = 'http://192.168.1.13:8000';
                          ApiConfig.setBaseUrl(_urlController.text);
                          repo.refreshAll();
                        },
                      ),
                      ActionChip(
                        label: const Text('10.0.2.2:8000 (Emulator)', style: TextStyle(fontSize: 11)),
                        onPressed: () {
                          _urlController.text = 'http://10.0.2.2:8000';
                          ApiConfig.setBaseUrl(_urlController.text);
                          repo.refreshAll();
                        },
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: AppColors.surfaceElevatedDark,
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              repo.backendHealth.isReachable ? Icons.check_circle : Icons.cancel,
                              size: 16,
                              color: repo.backendHealth.isReachable ? AppColors.success : AppColors.error,
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                repo.backendHealth.isReachable
                                    ? 'Connected: ${repo.backendHealth.runtimeIdentity} (v${repo.backendHealth.version})'
                                    : 'Disconnected: ${repo.backendHealth.error ?? "Unreachable"}',
                                style: TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.bold,
                                  color: repo.backendHealth.isReachable ? AppColors.success : AppColors.error,
                                ),
                              ),
                            ),
                          ],
                        ),
                        if (repo.backendHealth.isReachable) ...[
                          const SizedBox(height: 4),
                          Text(
                            'Model Loaded: ${repo.backendHealth.modelLoaded ? "YES (Ready)" : "NO (${repo.backendHealth.modelState})"}'
                            ' • Inference Supported: ${repo.backendHealth.inferenceSupported}',
                            style: const TextStyle(fontSize: 11, color: AppColors.neutral),
                          ),
                        ],
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    alignment: WrapAlignment.spaceBetween,
                    crossAxisAlignment: WrapCrossAlignment.center,
                    children: [
                      const Text(
                        '100% Local Inference • No External Telemetry',
                        style: TextStyle(fontSize: 11, color: AppColors.success),
                      ),
                      OutlinedButton(
                        onPressed: () {
                          ApiConfig.setBaseUrl(_urlController.text.trim());
                          repo.refreshAll();
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('Endpoint updated & health queried.')),
                          );
                        },
                        child: const Text('TEST & UPDATE ENDPOINT'),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // Architecture Invariant Diagnostics
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: AppColors.surfaceDark,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: AppColors.borderDark),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'RESEARCH ARCHITECTURE INVARIANTS',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.8,
                      color: AppColors.neutral,
                    ),
                  ),
                  const SizedBox(height: 12),
                  _specRow('Model Identifier', 'Qwen/Qwen2.5-1.5B-Instruct'),
                  _specRow('Exact Source Git Revision', '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
                  _specRow('Total Parameters', '1,543,714,304'),
                  _specRow('Transformer Layer Parameters', '1,310,339,072 (28 layers)'),
                  _specRow('Parameters Per Layer', '46,797,824'),
                  _specRow('BF16 Storage Per Layer', '89.26 MB (93,595,648 bytes)'),
                  _specRow('Residency Active Ratio', '1 / 28 (3.57% of transformer parameters)'),
                  _specRow('Attention Structure', '12 Attention Heads / 2 Key-Value Heads (GQA)'),
                  _specRow('Hidden / Intermediate Size', '1536 / 8960'),
                  _specRow('Vocabulary Size', '151,936 tokens (Tied embeddings)'),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _specRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: LayoutBuilder(
        builder: (context, constraints) {
          if (constraints.maxWidth < 360) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label, style: const TextStyle(fontSize: 12, color: AppColors.neutral)),
                const SizedBox(height: 2),
                Text(
                  value,
                  style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: Colors.white),
                ),
              ],
            );
          }
          return Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Flexible(child: Text(label, style: const TextStyle(fontSize: 12, color: AppColors.neutral))),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  value,
                  textAlign: TextAlign.end,
                  style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: Colors.white),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}
