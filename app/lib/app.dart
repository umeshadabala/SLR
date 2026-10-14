import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'core/theme/app_theme.dart';
import 'data/repositories/slr_repository.dart';
import 'features/benchmarks/benchmarks_screen.dart';
import 'features/dashboard/dashboard_screen.dart';
import 'features/experiments/experiments_screen.dart';
import 'features/inference/inference_screen.dart';
import 'features/models/models_screen.dart';
import 'features/residency/residency_screen.dart';
import 'features/settings/settings_screen.dart';

class SlrApp extends StatefulWidget {
  const SlrApp({super.key});

  @override
  State<SlrApp> createState() => _SlrAppState();
}

class _SlrAppState extends State<SlrApp> {
  int _currentIndex = 0;

  final List<Widget> _screens = const [
    DashboardScreen(),
    ModelsScreen(),
    InferenceScreen(),
    ResidencyScreen(),
    BenchmarksScreen(),
    ExperimentsScreen(),
    SettingsScreen(),
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final repo = context.read<SlrRepository>();
      repo.refreshAll();
      repo.startTelemetryPolling();
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Selective Layer Residency (EXP-07)',
      theme: AppTheme.darkTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: ThemeMode.dark,
      debugShowCheckedModeBanner: false,
      home: LayoutBuilder(
        builder: (context, constraints) {
          final isWide = constraints.maxWidth > 720;
          if (isWide) {
            return Scaffold(
              body: Row(
                children: [
                  NavigationRail(
                    selectedIndex: _currentIndex,
                    onDestinationSelected: (idx) => setState(() => _currentIndex = idx),
                    extended: constraints.maxWidth > 1100,
                    leading: Padding(
                      padding: const EdgeInsets.symmetric(vertical: 20),
                      child: Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: Theme.of(context).colorScheme.primary.withValues(alpha: 0.2),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Icon(
                              Icons.auto_awesome,
                              color: Theme.of(context).colorScheme.primary,
                              size: 20,
                            ),
                          ),
                          if (constraints.maxWidth > 1100) ...[
                            const SizedBox(width: 10),
                            const Text(
                              'EXP-07 SLR',
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                letterSpacing: 1.0,
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                    destinations: const [
                      NavigationRailDestination(
                        icon: Icon(Icons.dashboard_outlined),
                        selectedIcon: Icon(Icons.dashboard),
                        label: Text('Dashboard'),
                      ),
                      NavigationRailDestination(
                        icon: Icon(Icons.folder_zip_outlined),
                        selectedIcon: Icon(Icons.folder_zip),
                        label: Text('Models'),
                      ),
                      NavigationRailDestination(
                        icon: Icon(Icons.psychology_outlined),
                        selectedIcon: Icon(Icons.psychology),
                        label: Text('Inference'),
                      ),
                      NavigationRailDestination(
                        icon: Icon(Icons.layers_outlined),
                        selectedIcon: Icon(Icons.layers),
                        label: Text('Residency'),
                      ),
                      NavigationRailDestination(
                        icon: Icon(Icons.analytics_outlined),
                        selectedIcon: Icon(Icons.analytics),
                        label: Text('Benchmarks'),
                      ),
                      NavigationRailDestination(
                        icon: Icon(Icons.science_outlined),
                        selectedIcon: Icon(Icons.science),
                        label: Text('Experiments'),
                      ),
                      NavigationRailDestination(
                        icon: Icon(Icons.settings_outlined),
                        selectedIcon: Icon(Icons.settings),
                        label: Text('Settings'),
                      ),
                    ],
                  ),
                  const VerticalDivider(thickness: 1, width: 1),
                  Expanded(child: _screens[_currentIndex]),
                ],
              ),
            );
          } else {
            // Adaptive mobile layout: 5 primary tabs with drawer for all destinations
            return Scaffold(
              drawer: Drawer(
                child: ListView(
                  padding: EdgeInsets.zero,
                  children: [
                    DrawerHeader(
                      decoration: BoxDecoration(
                        color: Theme.of(context).colorScheme.surface,
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.auto_awesome,
                            color: Theme.of(context).colorScheme.primary,
                            size: 28,
                          ),
                          const SizedBox(height: 10),
                          const Text(
                            'EXP-07 SLR RESEARCH',
                            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                          ),
                          const SizedBox(height: 4),
                          const Text(
                            'Selective Layer Residency On-Device',
                            style: TextStyle(fontSize: 11, color: Colors.white70),
                          ),
                        ],
                      ),
                    ),
                    _drawerTile(0, 'Dashboard', Icons.dashboard),
                    _drawerTile(1, 'Models & Checkpoints', Icons.folder_zip),
                    _drawerTile(2, 'Inference', Icons.psychology),
                    _drawerTile(3, 'Residency Telemetry', Icons.layers),
                    _drawerTile(4, 'Benchmarks', Icons.analytics),
                    _drawerTile(5, 'Experiments', Icons.science),
                    _drawerTile(6, 'Settings', Icons.settings),
                  ],
                ),
              ),
              body: _screens[_currentIndex],
              bottomNavigationBar: NavigationBar(
                selectedIndex: _currentIndex > 3 ? 4 : _currentIndex,
                onDestinationSelected: (idx) {
                  if (idx == 4) {
                    _showMoreMenu(context);
                  } else {
                    setState(() => _currentIndex = idx);
                  }
                },
                destinations: [
                  const NavigationDestination(icon: Icon(Icons.dashboard), label: 'Dashboard'),
                  const NavigationDestination(icon: Icon(Icons.folder_zip), label: 'Models'),
                  const NavigationDestination(icon: Icon(Icons.psychology), label: 'Inference'),
                  const NavigationDestination(icon: Icon(Icons.layers), label: 'Residency'),
                  NavigationDestination(
                    icon: Icon(_currentIndex > 3 ? Icons.menu_open : Icons.more_horiz),
                    label: _currentIndex > 3 ? _getDestinationName(_currentIndex) : 'More',
                  ),
                ],
              ),
            );
          }
        },
      ),
    );
  }

  String _getDestinationName(int idx) {
    switch (idx) {
      case 4:
        return 'Benchmarks';
      case 5:
        return 'Experiments';
      case 6:
        return 'Settings';
      default:
        return 'More';
    }
  }

  Widget _drawerTile(int index, String title, IconData icon) {
    final isSelected = _currentIndex == index;
    return ListTile(
      leading: Icon(icon, color: isSelected ? Theme.of(context).colorScheme.primary : null),
      title: Text(
        title,
        style: TextStyle(
          fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          color: isSelected ? Theme.of(context).colorScheme.primary : null,
        ),
      ),
      selected: isSelected,
      onTap: () {
        Navigator.pop(context);
        setState(() => _currentIndex = index);
      },
    );
  }

  void _showMoreMenu(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Theme.of(context).colorScheme.surface,
      builder: (ctx) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.analytics),
                title: const Text('Benchmarks'),
                onTap: () {
                  Navigator.pop(ctx);
                  setState(() => _currentIndex = 4);
                },
              ),
              ListTile(
                leading: const Icon(Icons.science),
                title: const Text('Experiments'),
                onTap: () {
                  Navigator.pop(ctx);
                  setState(() => _currentIndex = 5);
                },
              ),
              ListTile(
                leading: const Icon(Icons.settings),
                title: const Text('Settings & Configuration'),
                onTap: () {
                  Navigator.pop(ctx);
                  setState(() => _currentIndex = 6);
                },
              ),
            ],
          ),
        );
      },
    );
  }
}
