import 'package:flutter/material.dart';
import '../../core/theme/app_colors.dart';
import '../../data/models/generation_job.dart';

class LayerWaterfallChart extends StatelessWidget {
  final int totalLayers;
  final int? activeLayerIdx;
  final List<LayerTiming> layerTimings;

  const LayerWaterfallChart({
    super.key,
    this.totalLayers = 28,
    this.activeLayerIdx,
    this.layerTimings = const [],
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isDark ? AppColors.surfaceDark : AppColors.surfaceLight,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: isDark ? AppColors.borderDark : AppColors.borderLight,
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
              const Text(
                'RESIDENCY MATRIX (28 LAYERS)',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 0.5,
                ),
              ),
              Wrap(
                spacing: 8,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  _legendDot(AppColors.primary, 'Active (CPU)'),
                  _legendDot(AppColors.borderDark, 'Dormant (META)'),
                ],
              ),
            ],
          ),
          const SizedBox(height: 16),
          LayoutBuilder(
            builder: (context, constraints) {
              final columns = constraints.maxWidth < 340
                  ? 4
                  : (constraints.maxWidth < 480 ? 5 : 7);
              return GridView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: columns,
                  crossAxisSpacing: 6,
                  mainAxisSpacing: 6,
                  mainAxisExtent: 44,
                ),
                itemCount: totalLayers,
                itemBuilder: (context, index) {
                  final isActive = activeLayerIdx == index;
                  final timing = layerTimings.firstWhere(
                    (t) => t.layerIdx == index,
                    orElse: () => LayerTiming(
                      layerIdx: index,
                      loadSec: 0,
                      execSec: 0,
                      releaseSec: 0,
                      activeParams: 0,
                      activeLayers: 0,
                      rssGb: 0,
                      kvCacheLen: 0,
                    ),
                  );

                  return Container(
                    padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 3),
                    decoration: BoxDecoration(
                      color: isActive
                          ? AppColors.primary.withValues(alpha: 0.2)
                          : (isDark ? AppColors.surfaceElevatedDark : AppColors.surfaceElevatedLight),
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(
                        color: isActive
                            ? AppColors.primary
                            : (isDark ? AppColors.borderDark : AppColors.borderLight),
                        width: isActive ? 1.5 : 1.0,
                      ),
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Flexible(
                              child: Text(
                                'L$index',
                                maxLines: 1,
                                style: TextStyle(
                                  fontSize: 10,
                                  fontWeight: FontWeight.bold,
                                  color: isActive ? AppColors.primary : (isDark ? Colors.white70 : Colors.black87),
                                ),
                              ),
                            ),
                            Text(
                              isActive ? 'CPU' : 'META',
                              style: TextStyle(
                                fontSize: 8.5,
                                fontWeight: FontWeight.w600,
                                color: isActive ? AppColors.primary : AppColors.neutral,
                              ),
                            ),
                          ],
                        ),
                    if (timing.loadSec > 0 || timing.execSec > 0) ...[
                      const SizedBox(height: 2),
                      Text(
                        '${((timing.loadSec + timing.execSec) * 1000).toStringAsFixed(1)}ms',
                        style: const TextStyle(
                          fontSize: 9,
                          color: AppColors.neutral,
                        ),
                      ),
                    ],
                  ],
                ),
              );
            },
          );
        },
      ),
    ],
  ),
);
  }

  Widget _legendDot(Color color, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(width: 4),
        Text(
          label,
          style: const TextStyle(fontSize: 10, color: AppColors.neutral),
        ),
      ],
    );
  }
}
