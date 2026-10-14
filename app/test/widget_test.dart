import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:exp07_slr/app.dart';
import 'package:exp07_slr/data/models/model_status.dart';
import 'package:exp07_slr/data/repositories/slr_repository.dart';
import 'package:exp07_slr/shared/widgets/metric_card.dart';
import 'package:exp07_slr/shared/widgets/status_badge.dart';

void main() {
  testWidgets('MetricCard displays title, value and unit', (WidgetTester tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: MetricCard(
            title: 'Residency Factor',
            value: '28×',
            unit: 'Layers',
            subtitle: '1 active layer materialized',
          ),
        ),
      ),
    );

    expect(find.text('RESIDENCY FACTOR'), findsOneWidget);
    expect(find.text('28×'), findsOneWidget);
    expect(find.text('Layers'), findsOneWidget);
  });

  testWidgets('StatusBadge shows correct text and simulation indicator', (WidgetTester tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: StatusBadge(
            status: 'ready',
            isSimulated: true,
          ),
        ),
      ),
    );

    expect(find.text('READY'), findsOneWidget);
    expect(find.text('SIMULATION'), findsOneWidget);
  });

  test('ModelStatus offline factory produces correct truthful values', () {
    final status = ModelStatus.offline(error: 'Connection refused');
    expect(status.isLoaded, false);
    expect(status.state, 'offline');
    expect(status.isSimulated, false);
    expect(status.totalParameters, 1543714304);
    expect(status.activeParameters, 0);
    expect(status.activeLayers, 0);
    expect(status.totalLayers, 28);
    expect(status.layerSizeMb, 89.26);
    expect(status.errorMessage, 'Connection refused');
  });

  testWidgets('SlrApp mounts successfully in offline state', (WidgetTester tester) async {
    await tester.pumpWidget(
      MultiProvider(
        providers: [
          ChangeNotifierProvider(
            create: (_) => SlrRepository(),
          ),
        ],
        child: const SlrApp(),
      ),
    );

    await tester.pump();
    expect(find.text('SLR RESEARCH DASHBOARD — EXP-07'), findsOneWidget);
  });
}
