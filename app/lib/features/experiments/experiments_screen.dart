import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import '../../core/theme/app_colors.dart';
import '../../data/models/experiment_data.dart';
import '../../data/repositories/slr_repository.dart';
import '../../shared/widgets/status_badge.dart';

class ExperimentsScreen extends StatelessWidget {
  const ExperimentsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final repo = context.watch<SlrRepository>();
    final experiments = repo.experiments;

    return Scaffold(
      appBar: AppBar(
        title: const Text('RESEARCH EXPERIMENTS', overflow: TextOverflow.ellipsis),
        actions: [
          Builder(
            builder: (context) {
              final isNarrow = MediaQuery.of(context).size.width < 450;
              if (isNarrow) {
                return IconButton(
                  tooltip: 'Export JSON',
                  icon: const Icon(Icons.download, color: AppColors.primary),
                  onPressed: () => _exportJson(context, repo),
                );
              }
              return Padding(
                padding: const EdgeInsets.only(right: 12),
                child: ElevatedButton.icon(
                  onPressed: () => _exportJson(context, repo),
                  icon: const Icon(Icons.download, size: 16),
                  label: const Text('EXPORT JSON'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: Colors.white,
                  ),
                ),
              );
            },
          ),
        ],
      ),
      body: experiments.isEmpty
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.science_outlined, size: 48, color: AppColors.neutral),
                  const SizedBox(height: 12),
                  const Text(
                    'No experiments recorded yet.',
                    style: TextStyle(color: AppColors.neutral, fontSize: 14),
                  ),
                  const SizedBox(height: 12),
                  OutlinedButton.icon(
                    onPressed: () => repo.refreshAll(),
                    icon: const Icon(Icons.refresh, size: 16),
                    label: const Text('REFRESH'),
                  ),
                ],
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(20),
              itemCount: experiments.length,
              itemBuilder: (context, index) {
                final exp = experiments[index];
                return _experimentCard(exp);
              },
            ),
    );
  }

  Future<void> _exportJson(BuildContext context, SlrRepository repo) async {
    try {
      final jsonStr = await repo.exportResultsJson();
      await Clipboard.setData(ClipboardData(text: jsonStr));
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Full results JSON copied to clipboard!'),
            backgroundColor: AppColors.success,
          ),
        );
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Export failed: $e'),
            backgroundColor: AppColors.error,
          ),
        );
      }
    }
  }

  Widget _experimentCard(ExperimentData exp) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.surfaceDark,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppColors.borderDark),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: AppColors.primary.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        exp.experimentId,
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: AppColors.primary,
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        exp.title,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              StatusBadge(status: exp.status),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            exp.description,
            style: const TextStyle(fontSize: 12, color: Colors.white70, height: 1.4),
          ),
          const Divider(height: 24),
          Wrap(
            alignment: WrapAlignment.spaceBetween,
            runSpacing: 4,
            spacing: 8,
            children: [
              Text(
                'Date: ${exp.date}',
                style: const TextStyle(fontSize: 11, color: AppColors.neutral),
              ),
              Text(
                '${exp.artifactsCount} Artifacts • ${exp.benchmarks.length} Benchmarks',
                style: const TextStyle(fontSize: 11, color: AppColors.neutral),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
