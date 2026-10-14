import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'app.dart';
import 'core/models/model_manager.dart';
import 'data/api/slr_api_client.dart';
import 'data/repositories/slr_repository.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(
          create: (_) => SlrRepository(api: SlrApiClient()),
        ),
        ChangeNotifierProvider(
          create: (_) => ModelManager(),
        ),
      ],
      child: const SlrApp(),
    ),
  );
}
