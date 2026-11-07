import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/config/api_config.dart';
import '../../core/native/android_bridge.dart';
import '../../core/theme/app_colors.dart';
import '../../data/models/chat_message.dart';
import '../../data/models/system_status.dart';
import '../../data/repositories/slr_repository.dart';
import '../../shared/widgets/error_banner.dart';

class InferenceScreen extends StatefulWidget {
  const InferenceScreen({super.key});

  @override
  State<InferenceScreen> createState() => _InferenceScreenState();
}

class _InferenceScreenState extends State<InferenceScreen> {
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  bool _showSettings = false;
  final Set<String> _expandedTelemetryMsgIds = {};

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _handleSend(SlrRepository repo) {
    final text = _textController.text.trim();
    if (text.isEmpty || repo.isGenerating || !repo.systemStatus.canSendInference) return;
    _textController.clear();
    repo.sendChatMessage(text);
    _scrollToBottom();
  }

  @override
  Widget build(BuildContext context) {
    final repo = context.watch<SlrRepository>();
    final sysStatus = repo.systemStatus;
    final messages = repo.chatMessages;

    // Auto scroll when actively generating
    if (repo.isGenerating) {
      _scrollToBottom();
    }

    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'SLR CHAT',
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(width: 8),
                InkWell(
                  onTap: () {
                    final nextMode = repo.isAndroidNativeMode
                        ? ExecutionMode.macHosted
                        : ExecutionMode.androidNative;
                    repo.setExecutionMode(nextMode);
                  },
                  borderRadius: BorderRadius.circular(4),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1.5),
                    decoration: BoxDecoration(
                      color: repo.isAndroidNativeMode
                          ? AppColors.success.withValues(alpha: 0.2)
                          : (repo.isConnected
                              ? AppColors.primary.withValues(alpha: 0.2)
                              : AppColors.error.withValues(alpha: 0.2)),
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(
                        color: repo.isAndroidNativeMode
                            ? AppColors.success
                            : (repo.isConnected ? AppColors.primary : AppColors.error),
                        width: 0.8,
                      ),
                    ),
                    child: Text(
                      repo.isAndroidNativeMode
                          ? 'ANDROID-NATIVE INFERENCE'
                          : (repo.isConnected
                              ? 'MAC-HOSTED SLR INFERENCE'
                              : 'BACKEND DISCONNECTED'),
                      style: TextStyle(
                        fontSize: 8,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 0.4,
                        color: repo.isAndroidNativeMode
                            ? AppColors.success
                            : (repo.isConnected ? AppColors.primary : AppColors.error),
                      ),
                    ),
                  ),
                ),
              ],
            ),
            Text(
              sysStatus.label,
              style: TextStyle(fontSize: 9, fontWeight: FontWeight.bold, color: sysStatus.color),
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: Icon(
              _showSettings ? Icons.tune : Icons.tune_outlined,
              color: _showSettings ? AppColors.primary : Colors.white,
              size: 20,
            ),
            tooltip: 'Generation Settings',
            onPressed: () => setState(() => _showSettings = !_showSettings),
          ),
          IconButton(
            icon: const Icon(Icons.refresh, size: 20),
            tooltip: 'New Chat / Clear Context',
            onPressed: messages.isEmpty ? null : () => repo.clearChat(),
          ),
        ],
      ),
      body: Column(
        children: [
          // 1. Detailed System Status Banner (Only for Mac-Hosted mode when disconnected)
          if (!repo.isAndroidNativeMode && sysStatus == SlrSystemStatus.backendUnreachable)
            Padding(
              padding: const EdgeInsets.all(12),
              child: Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: AppColors.error.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: AppColors.error.withValues(alpha: 0.4)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.link_off, color: AppColors.error, size: 20),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'SLR Backend Unreachable (${ApiConfig.baseUrl})',
                            style: const TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.bold,
                              color: AppColors.error,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      AndroidNativeBridge.isAndroid
                          ? 'The physical phone cannot connect to the Python inference server. '
                              'Run "adb reverse tcp:8000 tcp:8000" on your Mac over USB, or change the endpoint in Settings.'
                          : 'Cannot connect to Python server on ${ApiConfig.baseUrl}. Please verify the server is running.',
                      style: const TextStyle(fontSize: 12, color: Colors.white70, height: 1.3),
                    ),
                    const SizedBox(height: 10),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        ElevatedButton.icon(
                          onPressed: () => repo.retryConnection(),
                          icon: const Icon(Icons.refresh, size: 14),
                          label: const Text('RETRY CONNECTION', style: TextStyle(fontSize: 12)),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppColors.primary,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            )
          else if (!sysStatus.canSendInference && sysStatus != SlrSystemStatus.inferenceRunning)
            Padding(
              padding: const EdgeInsets.all(12),
              child: ErrorBanner(
                title: 'Model Not Ready: ${sysStatus.label}',
                message:
                    'Inference requires verified weights on the active engine. Current state is "${repo.modelStatus.state}". '
                    'Per SLR research standards, token generation is never simulated.',
                guidance: repo.modelStatus.checkpointGuidance ??
                    'Load model.safetensors via Settings or verify backend loaded status.',
              ),
            ),

          // Collapsible Hyperparameter Drawer
          if (_showSettings) _buildSettingsPanel(repo),

          // Main Chat Message Area
          Expanded(
            child: messages.isEmpty
                ? _buildEmptyState(repo)
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                    itemCount: messages.length,
                    itemBuilder: (context, index) {
                      final msg = messages[index];
                      return _buildMessageBubble(msg, repo);
                    },
                  ),
          ),

          // Global Error banner if present
          if (repo.lastError != null)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              color: AppColors.error.withValues(alpha: 0.15),
              child: Row(
                children: [
                  const Icon(Icons.error_outline, size: 16, color: AppColors.error),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      repo.lastError!,
                      style: const TextStyle(fontSize: 12, color: AppColors.error),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  TextButton(
                    onPressed: () => repo.retryConnection(),
                    child: const Text('RETRY', style: TextStyle(fontSize: 11, color: AppColors.primary)),
                  ),
                ],
              ),
            ),

          // Bottom Input Composer
          _buildComposer(repo),
        ],
      ),
    );
  }

  Widget _buildSettingsPanel(SlrRepository repo) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: const BoxDecoration(
        color: AppColors.surfaceElevatedDark,
        border: Border(bottom: BorderSide(color: AppColors.borderDark)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            alignment: WrapAlignment.spaceBetween,
            crossAxisAlignment: WrapCrossAlignment.center,
            spacing: 8,
            runSpacing: 4,
            children: [
              const Text(
                'GENERATION HYPERPARAMETERS',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 0.8,
                  color: AppColors.neutral,
                ),
              ),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text('Greedy Decoding', style: TextStyle(fontSize: 12)),
                  Switch(
                    value: repo.greedyDecoding,
                    activeThumbColor: AppColors.primary,
                    onChanged: (val) => repo.setGreedyDecoding(val),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              const Text('Max New Tokens:', style: TextStyle(fontSize: 12, color: AppColors.neutral)),
              ...[16, 32, 64, 128, 256].map((tokens) {
                final isSelected = repo.maxNewTokens == tokens;
                return ChoiceChip(
                  label: Text('$tokens'),
                  selected: isSelected,
                  selectedColor: AppColors.primary,
                  backgroundColor: AppColors.surfaceDark,
                  labelStyle: TextStyle(
                    fontSize: 11,
                    fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                    color: isSelected ? Colors.white : AppColors.neutral,
                  ),
                  onSelected: (_) => repo.setMaxNewTokens(tokens),
                );
              }),
            ],
          ),
          if (!repo.greedyDecoding) ...[
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Temperature: ${repo.temperature.toStringAsFixed(2)}',
                        style: const TextStyle(fontSize: 11, color: AppColors.neutral),
                      ),
                      Slider(
                        value: repo.temperature,
                        min: 0.1,
                        max: 1.5,
                        divisions: 14,
                        activeColor: AppColors.primary,
                        onChanged: (v) => repo.setTemperature(v),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Top-P (Nucleus): ${repo.topP.toStringAsFixed(2)}',
                        style: const TextStyle(fontSize: 11, color: AppColors.neutral),
                      ),
                      Slider(
                        value: repo.topP,
                        min: 0.1,
                        max: 1.0,
                        divisions: 9,
                        activeColor: AppColors.accent,
                        onChanged: (v) => repo.setTopP(v),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildEmptyState(SlrRepository repo) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: AppColors.primary.withValues(alpha: 0.12),
                border: Border.all(color: AppColors.primary.withValues(alpha: 0.3)),
              ),
              child: const Icon(Icons.memory, size: 40, color: AppColors.primary),
            ),
            const SizedBox(height: 16),
            const Text(
              'Selective Layer Residency Engine',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              'Qwen2.5-1.5B-Instruct • Strict 1/28 Layer Residency (46.8M active params)\n'
              'Prefill & autoregressive tokens are computed sequentially with real streaming SSE.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 12, color: AppColors.neutral, height: 1.4),
            ),
            const SizedBox(height: 24),
            const Text(
              'PROMPT SUGGESTIONS',
              style: TextStyle(
                fontSize: 10,
                fontWeight: FontWeight.bold,
                letterSpacing: 0.8,
                color: AppColors.neutral,
              ),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              alignment: WrapAlignment.center,
              children: [
                _suggestionChip(repo, 'Count from 1 to 5.'),
                _suggestionChip(repo, 'What is a CPU cache line?'),
                _suggestionChip(repo, 'Explain Selective Layer Residency in 2 sentences.'),
                _suggestionChip(repo, 'Why is memory bandwidth a bottleneck for LLMs?'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _suggestionChip(SlrRepository repo, String text) {
    return ActionChip(
      label: Text(text, style: const TextStyle(fontSize: 12)),
      backgroundColor: AppColors.surfaceElevatedDark,
      side: const BorderSide(color: AppColors.borderDark),
      onPressed: repo.systemStatus.canSendInference
          ? () {
              _textController.text = text;
              _handleSend(repo);
            }
          : null,
    );
  }

  Widget _buildMessageBubble(ChatMessageItem msg, SlrRepository repo) {
    final isUser = msg.role == 'user';
    final isExpanded = _expandedTelemetryMsgIds.contains(msg.id);

    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        children: [
          if (!isUser) ...[
            CircleAvatar(
              radius: 14,
              backgroundColor: msg.hasError
                  ? AppColors.error.withValues(alpha: 0.2)
                  : AppColors.primary.withValues(alpha: 0.2),
              child: Icon(
                msg.hasError ? Icons.error_outline : Icons.smart_toy,
                size: 16,
                color: msg.hasError ? AppColors.error : AppColors.primary,
              ),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Container(
              constraints: const BoxConstraints(maxWidth: 620),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: isUser
                    ? AppColors.primary.withValues(alpha: 0.25)
                    : (msg.hasError
                        ? AppColors.error.withValues(alpha: 0.12)
                        : AppColors.surfaceDark),
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(12),
                  topRight: const Radius.circular(12),
                  bottomLeft: isUser ? const Radius.circular(12) : const Radius.circular(2),
                  bottomRight: isUser ? const Radius.circular(2) : const Radius.circular(12),
                ),
                border: Border.all(
                  color: isUser
                      ? AppColors.primary.withValues(alpha: 0.5)
                      : (msg.hasError
                          ? AppColors.error.withValues(alpha: 0.5)
                          : AppColors.borderDark),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        isUser
                            ? 'USER'
                            : (msg.hasError ? 'INFERENCE FAILED' : 'QWEN-2.5-1.5B (SLR)'),
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                          color: isUser
                              ? AppColors.primary
                              : (msg.hasError ? AppColors.error : AppColors.accent),
                          letterSpacing: 0.5,
                        ),
                      ),
                      if (msg.isGenerating) ...[
                        const SizedBox(width: 8),
                        const SizedBox(
                          width: 8,
                          height: 8,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: 6),

                  // Message Body: Error state vs Generating vs Completed
                  if (msg.hasError) ...[
                    Text(
                      msg.error ?? 'Unknown error occurred.',
                      style: const TextStyle(fontSize: 13, color: AppColors.error, height: 1.3),
                    ),
                    const SizedBox(height: 8),
                    ElevatedButton.icon(
                      onPressed: () => repo.retryFailedMessage(msg),
                      icon: const Icon(Icons.refresh, size: 14),
                      label: const Text('RETRY THIS PROMPT', style: TextStyle(fontSize: 11)),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.primary,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      ),
                    ),
                  ] else if (msg.isCancelled) ...[
                    const Text(
                      'Generation cancelled by user.',
                      style: TextStyle(fontSize: 13, fontStyle: FontStyle.italic, color: AppColors.neutral),
                    ),
                  ] else ...[
                    SelectableText.rich(
                      TextSpan(
                        text: msg.content.isEmpty && msg.isGenerating
                            ? 'Connecting & prefilling prompt...'
                            : msg.content,
                        style: TextStyle(
                          fontSize: 14,
                          color: msg.content.isEmpty && msg.isGenerating
                              ? AppColors.neutral
                              : Colors.white,
                          fontStyle: msg.content.isEmpty && msg.isGenerating
                              ? FontStyle.italic
                              : FontStyle.normal,
                        ),
                        children: [
                          if (msg.isGenerating && msg.content.isNotEmpty)
                            const TextSpan(
                              text: ' ▍',
                              style: TextStyle(
                                color: AppColors.primary,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                        ],
                      ),
                    ),
                  ],

                  // Per-message Telemetry Inspector Footer (ONLY on successful completed Assistant response)
                  if (!isUser &&
                      msg.status == ChatMessageStatus.completed &&
                      (msg.generatedTokens ?? 0) > 0) ...[
                    const SizedBox(height: 8),
                    const Divider(height: 12),
                    InkWell(
                      onTap: () {
                        setState(() {
                          if (isExpanded) {
                            _expandedTelemetryMsgIds.remove(msg.id);
                          } else {
                            _expandedTelemetryMsgIds.add(msg.id);
                          }
                        });
                      },
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Expanded(
                            child: Wrap(
                              spacing: 6,
                              runSpacing: 4,
                              children: [
                                _chipMetric(
                                  'TTFT',
                                  msg.modelTtftSec != null
                                      ? '${msg.modelTtftSec!.toStringAsFixed(2)}s'
                                      : (msg.timeToFirstTokenSec != null
                                          ? '${msg.timeToFirstTokenSec!.toStringAsFixed(2)}s'
                                          : '-'),
                                ),
                                _chipMetric(
                                  'Decode',
                                  msg.decodeThroughput != null
                                      ? (msg.decodeThroughput! < 0.1
                                          ? '${msg.decodeThroughput!.toStringAsFixed(2)} tok/s'
                                          : '${msg.decodeThroughput!.toStringAsFixed(1)} tok/s')
                                      : (msg.decodeThroughputReason ?? 'N/A'),
                                ),
                                _chipMetric(
                                  'Tokens',
                                  '${msg.generatedTokens} out',
                                ),
                              ],
                            ),
                          ),
                          Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                isExpanded ? 'Hide Details' : 'Inspect Run',
                                style: const TextStyle(fontSize: 10, color: AppColors.neutral),
                              ),
                              Icon(
                                isExpanded ? Icons.expand_less : Icons.expand_more,
                                size: 16,
                                color: AppColors.neutral,
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    if (isExpanded) ...[
                      const SizedBox(height: 8),
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: AppColors.surfaceElevatedDark,
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(color: AppColors.borderDark),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                              margin: const EdgeInsets.only(bottom: 6),
                              decoration: BoxDecoration(
                                color: AppColors.primary.withValues(alpha: 0.15),
                                borderRadius: BorderRadius.circular(4),
                                border: Border.all(color: AppColors.primary, width: 0.8),
                              ),
                              child: Text(
                                msg.executionMode,
                                style: const TextStyle(
                                  fontSize: 9,
                                  fontWeight: FontWeight.bold,
                                  color: AppColors.primary,
                                  letterSpacing: 0.5,
                                ),
                              ),
                            ),
                            _telemetryRow('Execution Location', msg.executionLocation),
                            _telemetryRow('Active Layer Residency', '1 / 28 Layers (46.8M active params)'),
                            const Divider(height: 10),
                            _telemetryRow('Model TTFT (Prefill + 1st)', msg.modelTtftSec != null ? '${msg.modelTtftSec!.toStringAsFixed(3)} s' : '-'),
                            _telemetryRow('Server TTFT (Receipt to 1st)', msg.timeToFirstTokenSec != null ? '${msg.timeToFirstTokenSec!.toStringAsFixed(3)} s' : '-'),
                            _telemetryRow('UI First-Visible Latency', msg.uiFirstVisibleLatencySec != null ? '${msg.uiFirstVisibleLatencySec!.toStringAsFixed(3)} s' : '-'),
                            _telemetryRow('Prefill Latency', msg.prefillLatencySec != null ? '${msg.prefillLatencySec!.toStringAsFixed(3)} s' : '-'),
                            _telemetryRow('1st Token Decode', msg.firstTokenDecodeSec != null ? '${msg.firstTokenDecodeSec!.toStringAsFixed(3)} s' : '-'),
                            _telemetryRow('Steady-State Decode Throughput', msg.decodeThroughput != null ? '${msg.decodeThroughput!.toStringAsFixed(2)} tok/s' : (msg.decodeThroughputReason ?? 'N/A')),
                            if (msg.interTokenLatencySec != null)
                              _telemetryRow('Inter-Token Latency', '${msg.interTokenLatencySec!.toStringAsFixed(3)} s/tok'),
                            _telemetryRow('End-to-End Throughput', msg.endToEndThroughput != null ? '${msg.endToEndThroughput!.toStringAsFixed(2)} tok/s' : '-'),
                            _telemetryRow('Client-Observed Throughput', msg.clientObservedThroughput != null ? '${msg.clientObservedThroughput!.toStringAsFixed(2)} tok/s' : '-'),
                            _telemetryRow('Total Request Latency', msg.totalGenerationSec != null ? '${msg.totalGenerationSec!.toStringAsFixed(2)} s' : '-'),
                            const Divider(height: 10),
                            _telemetryRow('Tokens (Prompt / Generated)', '${msg.promptTokens ?? "-"} in / ${msg.generatedTokens ?? "-"} out'),
                            _telemetryRow('Stop Reason', msg.stopReason ?? 'eos'),
                            _telemetryRow('Peak Process RSS', msg.peakRssGb != null ? '${msg.peakRssGb!.toStringAsFixed(3)} GB' : '-'),
                            if (msg.fullReportJson?['memory_measurements'] != null)
                              _telemetryRow('Memory Composition', 'Anon: ${(msg.fullReportJson!["memory_measurements"]["rss_anon_mb"] as num?)?.toDouble().toStringAsFixed(1) ?? "-"} MB | File: ${(msg.fullReportJson!["memory_measurements"]["rss_file_mb"] as num?)?.toDouble().toStringAsFixed(1) ?? "-"} MB'),
                            if (msg.fullReportJson?['acceleration_mode'] != null)
                              _telemetryRow('Acceleration Mode', msg.fullReportJson!['acceleration_mode']?.toString() ?? 'NATIVE'),
                          ],
                        ),
                      ),
                    ],
                  ],
                ],
              ),
            ),
          ),
          if (isUser) ...[
            const SizedBox(width: 8),
            CircleAvatar(
              radius: 14,
              backgroundColor: AppColors.surfaceElevatedDark,
              child: const Icon(Icons.person, size: 16, color: Colors.white70),
            ),
          ],
        ],
      ),
    );
  }

  Widget _chipMetric(String label, String value) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: AppColors.surfaceElevatedDark,
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: AppColors.borderDark),
      ),
      child: Text(
        '$label: $value',
        style: const TextStyle(fontSize: 10, color: AppColors.primary, fontWeight: FontWeight.bold),
      ),
    );
  }

  Widget _telemetryRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 11, color: AppColors.neutral)),
          const SizedBox(width: 8),
          Flexible(
            child: Text(
              value,
              textAlign: TextAlign.end,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildComposer(SlrRepository repo) {
    final canSend = !repo.isGenerating && repo.systemStatus.canSendInference;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: const BoxDecoration(
        color: AppColors.surfaceDark,
        border: Border(top: BorderSide(color: AppColors.borderDark)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Expanded(
            child: TextField(
              controller: _textController,
              minLines: 1,
              maxLines: 4,
              enabled: repo.systemStatus.canSendInference,
              style: const TextStyle(fontSize: 14),
              decoration: InputDecoration(
                hintText: repo.systemStatus.canSendInference
                    ? 'Message Qwen2.5-1.5B (SLR Engine)...'
                    : '${repo.systemStatus.label} — see banner above',
                hintStyle: const TextStyle(fontSize: 13, color: AppColors.neutral),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(20),
                  borderSide: const BorderSide(color: AppColors.borderDark),
                ),
                contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              ),
              onSubmitted: (_) => _handleSend(repo),
            ),
          ),
          const SizedBox(width: 8),
          if (repo.isGenerating)
            IconButton(
              icon: const Icon(Icons.stop_circle, color: AppColors.error, size: 36),
              tooltip: 'Stop Generation',
              onPressed: () => repo.cancelGeneration(),
            )
          else
            IconButton(
              icon: Icon(
                Icons.arrow_upward_rounded,
                color: canSend ? AppColors.primary : AppColors.neutral,
                size: 28,
              ),
              tooltip: 'Send Prompt',
              onPressed: canSend ? () => _handleSend(repo) : null,
            ),
        ],
      ),
    );
  }
}
