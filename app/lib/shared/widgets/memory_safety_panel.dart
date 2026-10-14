import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/theme/app_colors.dart';
import '../../data/repositories/slr_repository.dart';

class MemorySafetyPanel extends StatelessWidget {
  const MemorySafetyPanel({super.key});

  @override
  Widget build(BuildContext context) {
    final repo = context.watch<SlrRepository>();
    final safety = repo.safetyStatus ?? {};
    final tel = repo.telemetry;
    final diag = repo.androidDiagnostics ?? {};

    final state = safety['state'] as String? ?? 'NORMAL';
    final isAborted = state == 'ABORTED' || state == 'CLEANUP_FAILED';
    final isWarning = state == 'WARNING';
    final currentRssMb = (safety['current_rss_mb'] as num?)?.toDouble() ?? (tel.processRssGb * 1024.0);
    final peakRssMb = (safety['peak_rss_mb'] as num?)?.toDouble() ?? currentRssMb;
    final totalRamMb = (safety['total_ram_mb'] as num?)?.toDouble() ?? ((diag['totalPhysicalRamMb'] as num?)?.toDouble() ?? 0.0);
    final availRamMb = (safety['available_ram_mb'] as num?)?.toDouble() ?? 0.0;
    final limitMb = (safety['effective_limit_mb'] as num?)?.toDouble() ?? (totalRamMb * 0.75);
    final activeIdx = (safety['current_active_layer_idx'] as int?) ?? (tel.currentLayerIdx ?? -1);
    final kvLen = (safety['current_kv_cache_length'] as int?) ?? tel.kvCacheSequenceLength;
    final lastEvent = safety['last_event'] as String? ?? '';

    Color stateColor = AppColors.success;
    if (isWarning) stateColor = AppColors.warning;
    if (isAborted) stateColor = AppColors.error;

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.surfaceDark,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: isAborted ? AppColors.error : (isWarning ? AppColors.warning : AppColors.borderDark),
          width: isAborted || isWarning ? 2 : 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: 8,
            runSpacing: 6,
            alignment: WrapAlignment.spaceBetween,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              Wrap(
                spacing: 8,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  Icon(
                    Icons.shield_outlined,
                    color: stateColor,
                    size: 20,
                  ),
                  const Text(
                    'MEMORY SAFETY FUSE',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.5,
                    ),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: stateColor.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: stateColor.withValues(alpha: 0.5)),
                ),
                child: Text(
                  state,
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: stateColor,
                  ),
                ),
              ),
            ],
          ),

          if (isAborted || isWarning) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: (isAborted ? AppColors.error : AppColors.warning).withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: isAborted ? AppColors.error : AppColors.warning),
              ),
              child: Wrap(
                spacing: 10,
                runSpacing: 8,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(isAborted ? Icons.dangerous : Icons.warning_amber_rounded,
                          color: isAborted ? AppColors.error : AppColors.warning, size: 24),
                      const SizedBox(width: 8),
                      Text(
                        isAborted ? 'FUSE TRIPPED' : 'MEMORY WARNING',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: isAborted ? AppColors.error : AppColors.warning,
                        ),
                      ),
                    ],
                  ),
                  Text(
                    lastEvent.isNotEmpty ? lastEvent : 'Memory safety limit reached.',
                    style: const TextStyle(fontSize: 11, color: Colors.white70),
                  ),
                  if (isAborted)
                    ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.primary,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      ),
                      onPressed: () => repo.acknowledgeFuseReset(),
                      child: const Text('Reset Fuse', style: TextStyle(fontSize: 11)),
                    ),
                ],
              ),
            ),
          ],

          const Divider(height: 24),

          // Real metric grid
          LayoutBuilder(
            builder: (context, constraints) {
              final tileWidth = constraints.maxWidth < 340
                  ? constraints.maxWidth
                  : (constraints.maxWidth - 20) / 2;

              Widget tile(String label, String value, String sub, Color valColor) {
                return SizedBox(
                  width: tileWidth,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        label.toUpperCase(),
                        style: const TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: AppColors.neutral),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        value,
                        style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: valColor),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        sub,
                        style: const TextStyle(fontSize: 9, color: Colors.white38),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                );
              }

              return Wrap(
                spacing: 20,
                runSpacing: 14,
                children: [
                  tile('Process RSS', '${currentRssMb.toStringAsFixed(1)} MB', 'Kernel VmRSS (/proc/self/status)', AppColors.primary),
                  tile('Peak RSS', '${peakRssMb.toStringAsFixed(1)} MB', 'Maximum observed memory', Colors.white),
                  tile('Configured Limit', '${limitMb.toStringAsFixed(0)} MB', '75% physical RAM cap', AppColors.warning),
                  tile('Available System RAM', availRamMb > 0 ? '${availRamMb.toStringAsFixed(0)} MB' : 'Unavailable', 'ActivityManager memory', AppColors.accent),
                  tile('Total Device RAM', totalRamMb > 0 ? '${totalRamMb.toStringAsFixed(0)} MB' : 'Unavailable', 'Physical installed RAM', Colors.white70),
                  tile('Active Layers', '${tel.activeTransformerLayers} / 28', 'Strict invariant: max 1 active', tel.activeTransformerLayers > 1 ? AppColors.error : AppColors.success),
                  tile('Active Layer Params', '${(tel.activeTransformerParams / 1000000).toStringAsFixed(1)} M', '46.8M when computing, 0 on release', AppColors.accent),
                  tile('Current Layer Index', activeIdx >= 0 ? 'Layer $activeIdx' : 'Idle (Released)', 'Materialized compute layer', Colors.white),
                  tile('KV Cache Length', '$kvLen tokens', 'Autoregressive context depth', Colors.white),
                ],
              );
            },
          ),

          const SizedBox(height: 18),

          // Safe Benchmark and Single Layer Action row
          LayoutBuilder(
            builder: (context, constraints) {
              final isCompact = constraints.maxWidth < 420;
              final benchmarkButton = ElevatedButton.icon(
                icon: repo.isRunningBenchmark
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                      )
                    : const Icon(Icons.speed, size: 18),
                label: Text(repo.isRunningBenchmark ? 'Running Benchmark...' : 'Run Safe Benchmark (18 Tests)'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                ),
                onPressed: repo.isRunningBenchmark
                    ? null
                    : () async {
                        final report = await repo.runAndroidSafeBenchmark();
                        if (context.mounted && report != null) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text('Safe Benchmark completed: ${report['summary']?['final_status']} (${report['summary']?['passed']}/18 tests passed)'),
                              backgroundColor: AppColors.success,
                            ),
                          );
                        }
                      },
              );

              final singleLayerButton = OutlinedButton.icon(
                icon: const Icon(Icons.play_circle_outline, size: 18),
                label: const Text('Test Single Layer'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: Colors.white,
                  side: const BorderSide(color: AppColors.borderDark),
                  padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                ),
                onPressed: () async {
                  final res = await repo.executeSingleLayerOnDevice();
                  if (context.mounted) {
                    if (res != null) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text('Layer 0 executed in ${res['elapsedMs']}ms. Active params returned to 0.'),
                          backgroundColor: AppColors.success,
                        ),
                      );
                    } else if (repo.lastError != null) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text('Layer test: ${repo.lastError}'),
                          backgroundColor: AppColors.error,
                        ),
                      );
                    }
                  }
                },
              );

              if (isCompact) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    benchmarkButton,
                    const SizedBox(height: 10),
                    singleLayerButton,
                  ],
                );
              } else {
                return Row(
                  children: [
                    Expanded(child: benchmarkButton),
                    const SizedBox(width: 12),
                    singleLayerButton,
                  ],
                );
              }
            },
          ),
        ],
      ),
    );
  }
}
