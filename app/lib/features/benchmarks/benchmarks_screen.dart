import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/theme/app_colors.dart';
import '../../data/models/benchmark_data.dart';
import '../../data/repositories/slr_repository.dart';
import '../../shared/widgets/status_badge.dart';

class BenchmarksScreen extends StatefulWidget {
  const BenchmarksScreen({super.key});

  @override
  State<BenchmarksScreen> createState() => _BenchmarksScreenState();
}

class _BenchmarksScreenState extends State<BenchmarksScreen> {
  String _selectedCategory = 'All';

  final List<String> _categories = [
    'All',
    'Structural',
    'Numerical',
    'Generation',
    'Physical Device',
    'Safety',
  ];

  @override
  Widget build(BuildContext context) {
    final repo = context.watch<SlrRepository>();
    final allBenchmarks = repo.benchmarks;
    final filteredBenchmarks = _selectedCategory == 'All'
        ? allBenchmarks
        : allBenchmarks.where((b) => b.category == _selectedCategory).toList();

    return Scaffold(
      appBar: AppBar(
        title: const Text('RESEARCH BENCHMARK RECORDS'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, size: 20),
            onPressed: () => repo.refreshBenchmarks(),
            tooltip: 'Refresh Benchmarks',
          ),
        ],
      ),
      body: Column(
        children: [
          // Category selector chips
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            color: AppColors.surfaceDark,
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: _categories.map((cat) {
                  final isSelected = _selectedCategory == cat;
                  return Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: ChoiceChip(
                      label: Text(cat),
                      selected: isSelected,
                      selectedColor: AppColors.primary,
                      backgroundColor: AppColors.surfaceElevatedDark,
                      labelStyle: TextStyle(
                        fontSize: 12,
                        fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                        color: isSelected ? Colors.white : AppColors.neutral,
                      ),
                      onSelected: (val) {
                        if (val) setState(() => _selectedCategory = cat);
                      },
                    ),
                  );
                }).toList(),
              ),
            ),
          ),
          Expanded(
            child: filteredBenchmarks.isEmpty
                ? Center(
                    child: Text(
                      allBenchmarks.isEmpty
                          ? 'No benchmark records available'
                          : 'No benchmarks found for category: $_selectedCategory',
                      style: const TextStyle(color: AppColors.neutral),
                    ),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: filteredBenchmarks.length,
                    itemBuilder: (context, index) {
                      final bm = filteredBenchmarks[index];
                      return _benchmarkCard(bm);
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Widget _benchmarkCard(BenchmarkData bm) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
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
                child: Wrap(
                  spacing: 8,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    Text(
                      bm.experimentId,
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: AppColors.primary,
                        letterSpacing: 0.5,
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: AppColors.surfaceElevatedDark,
                        borderRadius: BorderRadius.circular(4),
                        border: Border.all(color: AppColors.borderDark),
                      ),
                      child: Text(
                        bm.category.toUpperCase(),
                        style: const TextStyle(
                          fontSize: 9,
                          fontWeight: FontWeight.bold,
                          color: AppColors.accent,
                        ),
                      ),
                    ),
                    Text(
                      bm.title,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              StatusBadge(status: bm.validationStatus, isSimulated: bm.isSimulated),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            '${bm.model} • ${bm.precision} • ${bm.device} (${bm.osPlatform})',
            style: const TextStyle(fontSize: 12, color: AppColors.neutral),
          ),
          const Divider(height: 20),
          Wrap(
            spacing: 16,
            runSpacing: 10,
            children: [
              _metricCol(
                'THROUGHPUT',
                bm.formattedThroughput,
                bm.isGenerationBenchmark ? AppColors.primary : AppColors.neutral,
              ),
              _metricCol(
                'TIME TO FIRST TOKEN',
                bm.timeToFirstTokenSec != null ? '${bm.timeToFirstTokenSec!.toStringAsFixed(3)} s' : 'N/A',
                Colors.white,
              ),
              _metricCol('TOTAL LATENCY', '${bm.totalGenerationSec.toStringAsFixed(3)} s', Colors.white),
              _metricCol('PEAK RSS', '${bm.peakRssGb.toStringAsFixed(3)} GB', AppColors.warning),
              _metricCol('ACTIVE RATIO', bm.activeTransformerParamsRatio, AppColors.accent),
            ],
          ),
          if (bm.notes.isNotEmpty) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: AppColors.surfaceElevatedDark,
                borderRadius: BorderRadius.circular(6),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: bm.notes
                    .map((n) => Padding(
                          padding: const EdgeInsets.only(bottom: 2),
                          child: Text(
                            '• $n',
                            style: const TextStyle(fontSize: 11, color: Colors.white70),
                          ),
                        ))
                    .toList(),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _metricCol(String label, String value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 9, fontWeight: FontWeight.bold, color: AppColors.neutral),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: color),
        ),
      ],
    );
  }
}
