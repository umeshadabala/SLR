import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:exp07_slr/app.dart';
import 'package:exp07_slr/core/models/model_manager.dart';
import 'package:exp07_slr/data/repositories/slr_repository.dart';
import 'package:exp07_slr/features/benchmarks/benchmarks_screen.dart';
import 'package:exp07_slr/features/dashboard/dashboard_screen.dart';
import 'package:exp07_slr/features/experiments/experiments_screen.dart';
import 'package:exp07_slr/features/inference/inference_screen.dart';
import 'package:exp07_slr/features/models/models_screen.dart';
import 'package:exp07_slr/features/residency/residency_screen.dart';
import 'package:exp07_slr/features/settings/settings_screen.dart';

void main() {
  const testMatrix = <String, Size>{
    'iPhone SE 1st gen': Size(320, 568),
    'Small Android Phone': Size(360, 640),
    'Modern iPhone (390x844)': Size(390, 844),
    'Modern Pixel/Galaxy (412x915)': Size(412, 915),
    'Tablet/Desktop (1024x768)': Size(1024, 768),
  };

  Widget buildTestScreen(Widget screen) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => SlrRepository()),
        ChangeNotifierProvider(create: (_) => ModelManager()),
      ],
      child: MaterialApp(
        theme: ThemeData.dark(),
        home: screen,
      ),
    );
  }

  for (final entry in testMatrix.entries) {
    final deviceName = entry.key;
    final size = entry.value;

    group('Responsive Screen Matrix — $deviceName (${size.width.toInt()}x${size.height.toInt()})', () {
      testWidgets('DashboardScreen renders cleanly', (tester) async {
        tester.view.physicalSize = size;
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);

        await tester.pumpWidget(buildTestScreen(const DashboardScreen()));
        await tester.pump();
        expect(tester.takeException(), isNull);
      });

      testWidgets('InferenceScreen renders cleanly', (tester) async {
        tester.view.physicalSize = size;
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);

        await tester.pumpWidget(buildTestScreen(const InferenceScreen()));
        await tester.pump();
        expect(tester.takeException(), isNull);
      });

      testWidgets('ResidencyScreen renders cleanly', (tester) async {
        tester.view.physicalSize = size;
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);

        await tester.pumpWidget(buildTestScreen(const ResidencyScreen()));
        await tester.pump();
        expect(tester.takeException(), isNull);
      });

      testWidgets('BenchmarksScreen renders cleanly', (tester) async {
        tester.view.physicalSize = size;
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);

        await tester.pumpWidget(buildTestScreen(const BenchmarksScreen()));
        await tester.pump();
        expect(tester.takeException(), isNull);
      });

      testWidgets('ModelsScreen renders cleanly', (tester) async {
        tester.view.physicalSize = size;
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);

        await tester.pumpWidget(buildTestScreen(const ModelsScreen()));
        await tester.pump();
        expect(tester.takeException(), isNull);
      });

      testWidgets('ExperimentsScreen renders cleanly', (tester) async {
        tester.view.physicalSize = size;
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);

        await tester.pumpWidget(buildTestScreen(const ExperimentsScreen()));
        await tester.pump();
        expect(tester.takeException(), isNull);
      });

      testWidgets('SettingsScreen renders cleanly', (tester) async {
        tester.view.physicalSize = size;
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);

        await tester.pumpWidget(buildTestScreen(const SettingsScreen()));
        await tester.pump();
        expect(tester.takeException(), isNull);
      });

      testWidgets('Full SlrApp navigation renders cleanly', (tester) async {
        tester.view.physicalSize = size;
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.resetPhysicalSize);

        await tester.pumpWidget(
          MultiProvider(
            providers: [
              ChangeNotifierProvider(create: (_) => SlrRepository()),
              ChangeNotifierProvider(create: (_) => ModelManager()),
            ],
            child: const SlrApp(),
          ),
        );
        await tester.pump();
        expect(tester.takeException(), isNull);
      });
    });
  }
}
