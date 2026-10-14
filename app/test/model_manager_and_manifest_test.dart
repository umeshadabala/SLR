import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:exp07_slr/core/models/model_manager.dart';
import 'package:exp07_slr/data/models/model_manifest.dart';

void main() {
  group('ModelManifest & Checkpoint Lifecycle Tests', () {
    test('Initializes with default Qwen2.5-1.5B-Instruct metadata', () {
      final manifest = ModelManifest.qwen25Default('/tmp/test_dir');

      expect(manifest.modelId, 'Qwen/Qwen2.5-1.5B-Instruct');
      expect(manifest.revision, '989aa7980e4cf806f80c7fef2b1adb7bc71aa306');
      expect(manifest.state, ModelLifecycleState.notDownloaded);
      expect(manifest.files.length, 7);

      final safetensors = manifest.files.firstWhere((f) => f.relativePath == 'model.safetensors');
      expect(safetensors.expectedSizeBytes, 3087467144);
      expect(safetensors.expectedSha256, 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee');
      expect(safetensors.isRequired, true);

      final configJson = manifest.files.firstWhere((f) => f.relativePath == 'config.json');
      expect(configJson.expectedSizeBytes, 660);
      expect(configJson.expectedSha256, '98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670');

      final tokenizerJson = manifest.files.firstWhere((f) => f.relativePath == 'tokenizer.json');
      expect(tokenizerJson.expectedSizeBytes, 7031645);
      expect(tokenizerJson.expectedSha256, 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539');
    });

    test('10 lifecycle states have accurate distinct human-readable labels', () {
      final expectedLabels = {
        ModelLifecycleState.notDownloaded: 'Not Downloaded',
        ModelLifecycleState.downloading: 'Downloading',
        ModelLifecycleState.downloadInterrupted: 'Download Interrupted',
        ModelLifecycleState.downloaded: 'Downloaded (Unverified)',
        ModelLifecycleState.integrityVerified: 'Integrity Verified',
        ModelLifecycleState.runtimeCompatibilityVerified: 'Runtime Compatible',
        ModelLifecycleState.readyToLoad: 'Ready to Load',
        ModelLifecycleState.loading: 'Loading',
        ModelLifecycleState.loaded: 'Loaded (Active)',
        ModelLifecycleState.failed: 'Failed',
      };

      for (final state in ModelLifecycleState.values) {
        expect(state.label, expectedLabels[state]);
      }
    });

    test('Serialization and deserialization preserve manifest fields and hashes', () {
      final original = ModelManifest.qwen25Default('/tmp/models');
      original.state = ModelLifecycleState.downloaded;
      original.files[0].actualSizeBytes = 660;
      original.files[0].isVerified = true;
      original.files[0].computedSha256 = original.files[0].expectedSha256;

      final map = original.toJson();
      final restored = ModelManifest.fromJson(map);

      expect(restored.modelId, original.modelId);
      expect(restored.revision, original.revision);
      expect(restored.state, ModelLifecycleState.downloaded);
      expect(restored.files[0].isVerified, true);
      expect(restored.files[0].computedSha256, original.files[0].expectedSha256);
    });

    test('Inspect local files detects absent files correctly', () async {
      final tempDir = Directory.systemTemp.createTempSync('slr_test_');
      addTearDown(() => tempDir.deleteSync(recursive: true));

      final mgr = ModelManager(baseStorageDir: tempDir.path);
      await mgr.inspectLocalFiles();

      expect(mgr.state, ModelLifecycleState.notDownloaded);
      expect(mgr.manifest.files.every((f) => f.actualSizeBytes == 0), true);
    });

    test('Inspect local files detects present files and updates sizes', () async {
      final tempDir = Directory.systemTemp.createTempSync('slr_test_');
      addTearDown(() => tempDir.deleteSync(recursive: true));

      final mgr = ModelManager(baseStorageDir: tempDir.path);
      final modelDir = Directory(mgr.manifest.targetDirectory)..createSync(recursive: true);

      // Create dummy config.json of 660 bytes
      final configFile = File('${modelDir.path}/config.json');
      configFile.writeAsBytesSync(List.filled(660, 0));

      await mgr.inspectLocalFiles();

      final configEntry = mgr.manifest.files.firstWhere((f) => f.relativePath == 'config.json');
      expect(configEntry.actualSizeBytes, 660);
      expect(mgr.state, ModelLifecycleState.downloadInterrupted);
    });

    test('Verify integrity fails when file hash does not match expected SHA-256', () async {
      final tempDir = Directory.systemTemp.createTempSync('slr_test_');
      addTearDown(() => tempDir.deleteSync(recursive: true));

      final mgr = ModelManager(baseStorageDir: tempDir.path);
      final modelDir = Directory(mgr.manifest.targetDirectory)..createSync(recursive: true);

      // Create fake file with mismatched content
      final configFile = File('${modelDir.path}/config.json');
      configFile.writeAsStringSync('corrupted content');

      final ok = await mgr.verifyIntegrity();
      expect(ok, false);
      expect(mgr.state, ModelLifecycleState.failed);
      expect(mgr.manifest.errorMessage?.toLowerCase(), contains('verification failed'));
    });
  });
}
