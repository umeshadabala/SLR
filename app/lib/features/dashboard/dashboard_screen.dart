import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/theme/app_colors.dart';
import '../../data/repositories/slr_repository.dart';
import '../../shared/widgets/error_banner.dart';
import '../../shared/widgets/memory_safety_panel.dart';
import '../../shared/widgets/metric_card.dart';
import '../../shared/widgets/status_badge.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final repo = context.watch<SlrRepository>();
    final status = repo.modelStatus;
    final tel = repo.telemetry;

    return Scaffold(
      appBar: AppBar(
        title: const Text('SLR RESEARCH DASHBOARD — EXP-07'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: StatusBadge(
              status: status.state,
              isSimulated: status.isSimulated,
            ),
          ),
          IconButton(
            icon: const Icon(Icons.refresh, size: 20),
            onPressed: () => repo.refreshAll(),
            tooltip: 'Refresh Status',
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: () => repo.refreshAll(),
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (status.state == 'checkpoint_missing' || status.state == 'offline') ...[
                ErrorBanner(
                  title: status.state == 'offline'
                      ? 'Inference Backend Offline'
                      : 'Model Checkpoint Not Found',
                  message: status.errorMessage ??
                      'The local Python service is listening, but no valid Qwen2.5-1.5B-Instruct checkpoint was found.',
                  guidance: status.checkpointGuidance ??
                      'Start the backend service with: ./scripts/start_backend.sh',
                ),
                const SizedBox(height: 20),
              ],

              const MemorySafetyPanel(),
              const SizedBox(height: 20),

              GridView(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
                  maxCrossAxisExtent: 320,
                  mainAxisExtent: 110,
                  crossAxisSpacing: 14,
                  mainAxisSpacing: 14,
                ),
                children: [
                  MetricCard(
                    title: 'Residency Factor',
                    value: '${status.totalLayers}×',
                    unit: 'Layers',
                    subtitle: '1 active layer materialized at a time',
                    icon: Icons.layers_outlined,
                    accentColor: AppColors.primary,
                  ),
                  MetricCard(
                    title: 'Active Layer RAM',
                    value: status.layerSizeMb > 0 ? status.layerSizeMb.toStringAsFixed(2) : '89.26',
                    unit: 'MB',
                    subtitle: '46.8M BF16 parameters per layer',
                    icon: Icons.memory,
                    accentColor: AppColors.accent,
                  ),
                  MetricCard(
                    title: 'Process RSS',
                    value: tel.processRssGb > 0
                        ? tel.processRssGb.toStringAsFixed(2)
                        : 'Unavailable',
                    unit: tel.processRssGb > 0 ? 'GB' : null,
                    subtitle: 'Operating system resident memory',
                    icon: Icons.pie_chart_outline,
                    accentColor: AppColors.warning,
                  ),
                  MetricCard(
                    title: 'KV Cache Sequence',
                    value: '${tel.kvCacheSequenceLength}',
                    unit: 'Tokens',
                    subtitle: 'DynamicCache autoregressive depth',
                    icon: Icons.alt_route,
                    accentColor: AppColors.success,
                  ),
                ],
              ),

              const SizedBox(height: 24),

              // Model Metadata Card
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
                    Wrap(
                      spacing: 8,
                      runSpacing: 4,
                      alignment: WrapAlignment.spaceBetween,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        const Text(
                          'MODEL UNDER EVALUATION',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 0.5,
                          ),
                        ),
                        Text(
                          status.targetRevision.length >= 8
                              ? status.targetRevision.substring(0, 8)
                              : status.targetRevision,
                          style: const TextStyle(
                            fontSize: 11,
                            fontFamily: 'monospace',
                            color: AppColors.neutral,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Text(
                      status.modelId,
                      style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Target Revision: ${status.targetRevision} • Tied Embeddings • ${status.totalLayers} Transformer Layers • BF16 Precision',
                      style: const TextStyle(fontSize: 12, color: Colors.white70, height: 1.4),
                    ),
                    const Divider(height: 24),
                    Wrap(
                      spacing: 24,
                      runSpacing: 12,
                      children: [
                        _infoPair('Total Parameters', status.totalParameters > 0 ? '${status.totalParameters}' : '1,543,714,304'),
                        _infoPair('Transformer Layers', '${status.totalLayers} layers (46.8M params each)'),
                        _infoPair('Engine Mode', status.engineType),
                      ],
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // Recent Real Benchmarks
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
                    Wrap(
                      spacing: 8,
                      runSpacing: 4,
                      alignment: WrapAlignment.spaceBetween,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        const Text(
                          'VALIDATED BENCHMARKS',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 0.5,
                          ),
                        ),
                        Text(
                          '${repo.benchmarks.length} RECORDS',
                          style: const TextStyle(fontSize: 11, color: AppColors.neutral),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    if (repo.benchmarks.isEmpty)
                      const Padding(
                        padding: EdgeInsets.symmetric(vertical: 16),
                        child: Text(
                          'Loading validated benchmarks from results database...',
                          style: TextStyle(color: AppColors.neutral, fontSize: 12),
                        ),
                      )
                    else
                      ListView.separated(
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        itemCount: repo.benchmarks.take(3).length,
                        separatorBuilder: (context, index) => const Divider(height: 16),
                        itemBuilder: (context, idx) {
                          final b = repo.benchmarks[idx];
                          return Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      '${b.experimentId}: ${b.title}',
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                      style: const TextStyle(
                                        fontSize: 13,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                    const SizedBox(height: 3),
                                    Text(
                                      '${b.precision} • ${b.device} • TTFT: ${b.timeToFirstTokenSec ?? '-'}s',
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                      style: const TextStyle(
                                        fontSize: 11,
                                        color: AppColors.neutral,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(width: 8),
                              Column(
                                crossAxisAlignment: CrossAxisAlignment.end,
                                children: [
                                  Text(
                                    b.formattedThroughput,
                                    style: const TextStyle(
                                      fontSize: 14,
                                      fontWeight: FontWeight.bold,
                                      color: AppColors.primary,
                                    ),
                                  ),
                                  Text(
                                    b.validationStatus,
                                    style: const TextStyle(
                                      fontSize: 10,
                                      color: AppColors.success,
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          );
                        },
                      ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _infoPair(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label.toUpperCase(),
          style: const TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.w600,
            color: AppColors.neutral,
          ),
        ),
        const SizedBox(height: 3),
        Text(
          value,
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w500,
            color: Colors.white,
          ),
        ),
      ],
    );
  }
}
