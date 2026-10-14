import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/theme/app_colors.dart';
import '../../data/repositories/slr_repository.dart';
import '../../shared/widgets/layer_waterfall_chart.dart';
import '../../shared/widgets/metric_card.dart';

class ResidencyScreen extends StatelessWidget {
  const ResidencyScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final repo = context.watch<SlrRepository>();
    final tel = repo.telemetry;
    final job = repo.currentJob;

    return Scaffold(
      appBar: AppBar(
        title: const Text('SELECTIVE LAYER RESIDENCY TELEMETRY'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, size: 20),
            onPressed: () => repo.refreshTelemetry(),
            tooltip: 'Poll Telemetry',
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
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
                  title: 'Active Layers',
                  value: '${tel.activeTransformerLayers}',
                  unit: '/ 28',
                  subtitle: 'Single-layer residency invariant',
                  accentColor: AppColors.primary,
                  icon: Icons.layers,
                ),
                MetricCard(
                  title: 'Active Parameters',
                  value: tel.activeTransformerParams > 0
                      ? '${(tel.activeTransformerParams / 1e6).toStringAsFixed(1)}M'
                      : '0',
                  unit: '/ 1.31B',
                  subtitle: '3.57% of transformer weights',
                  accentColor: AppColors.accent,
                  icon: Icons.compress,
                ),
                MetricCard(
                  title: 'Process RSS Memory',
                  value: tel.processRssGb > 0
                      ? tel.processRssGb.toStringAsFixed(2)
                      : 'Unavailable',
                  unit: tel.processRssGb > 0 ? 'GB' : null,
                  subtitle: 'Actual OS process footprint',
                  accentColor: AppColors.warning,
                  icon: Icons.storage,
                ),
                MetricCard(
                  title: 'KV Cache Sequence',
                  value: '${tel.kvCacheSequenceLength}',
                  unit: 'Tokens',
                  subtitle: 'DynamicCache cached states',
                  accentColor: AppColors.success,
                  icon: Icons.timeline,
                ),
              ],
            ),

            const SizedBox(height: 24),

            // Live Waterfall / Matrix Representation
            LayerWaterfallChart(
              totalLayers: 28,
              activeLayerIdx: tel.currentLayerIdx,
              layerTimings: job?.layerTimings ?? [],
            ),

            const SizedBox(height: 24),

            // Layer Lifecycle Accounting
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
                    'PER-LAYER LATENCY ACCOUNTING (MEASURED)',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.8,
                      color: AppColors.neutral,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 24,
                    runSpacing: 12,
                    alignment: WrapAlignment.spaceAround,
                    children: [
                      _latencyMetric(
                        'SAFEPTR LOAD',
                        tel.lastLayerLoadSec != null
                            ? '${(tel.lastLayerLoadSec! * 1000).toStringAsFixed(1)} ms'
                            : 'Unavailable',
                        AppColors.layerLoad,
                      ),
                      _latencyMetric(
                        'CPU FORWARD EXEC',
                        tel.lastLayerExecSec != null
                            ? '${(tel.lastLayerExecSec! * 1000).toStringAsFixed(1)} ms'
                            : 'Unavailable',
                        AppColors.layerExec,
                      ),
                      _latencyMetric(
                        'META RELEASE',
                        tel.lastLayerReleaseSec != null
                            ? '${(tel.lastLayerReleaseSec! * 1000).toStringAsFixed(1)} ms'
                            : 'Unavailable',
                        AppColors.layerRelease,
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _latencyMetric(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.bold,
            color: color,
            letterSpacing: 0.5,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          value,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
      ],
    );
  }
}
