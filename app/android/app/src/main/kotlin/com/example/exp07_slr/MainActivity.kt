package com.example.exp07_slr

import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread

class MainActivity : FlutterActivity() {

    private val CHANNEL = "com.example.exp07_slr/channel"

    private lateinit var diagnostics: DeviceDiagnostics
    private lateinit var safetyController: MemorySafetyController
    private var engine: AndroidSlrEngine? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        diagnostics = DeviceDiagnostics(applicationContext)
        safetyController = MemorySafetyController(applicationContext, diagnostics)
        safetyController.startMonitoring()

        val checkpointFile = findCheckpointFile()
        if (checkpointFile != null) {
            engine = AndroidSlrEngine(applicationContext, checkpointFile, safetyController)
        }

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "getDeviceDiagnostics" -> {
                    val info = diagnostics.getDeviceInfo()
                    val map = mapOf(
                        "manufacturer" to info.manufacturer,
                        "model" to info.model,
                        "androidVersion" to info.androidVersion,
                        "apiLevel" to info.apiLevel,
                        "supportedAbis" to info.supportedAbis,
                        "cpuCoreCount" to info.cpuCoreCount,
                        "totalPhysicalRamMb" to info.totalPhysicalRamMb,
                        "appVersion" to info.appVersion,
                        "nativeRuntimeVersion" to info.nativeRuntimeVersion
                    )
                    result.success(map)
                }

                "getMemoryMetrics" -> {
                    val m = diagnostics.getMemoryMetrics()
                    val map = mapOf(
                        "processRssMb" to m.processRssMb,
                        "processRssSource" to m.processRssSource,
                        "peakProcessRssMb" to m.peakProcessRssMb,
                        "processPssMb" to m.processPssMb,
                        "availableSystemRamMb" to m.availableSystemRamMb,
                        "totalPhysicalRamMb" to m.totalPhysicalRamMb,
                        "isLowMemory" to m.isLowMemory,
                        "systemMemoryThresholdMb" to m.systemMemoryThresholdMb,
                        "timestampMs" to m.timestampMs
                    )
                    result.success(map)
                }

                "getSafetyStatus" -> {
                    val json = safetyController.getStatusJson()
                    result.success(json.toString())
                }

                "configureSafetyPolicy" -> {
                    val maxPercent = call.argument<Double>("maxProcessRssPercent") ?: 75.0
                    val maxMb = call.argument<Double>("maxProcessRssMb")
                    val minAvailMb = call.argument<Double>("minAvailableSystemRamMb") ?: 256.0
                    val warnPercent = call.argument<Double>("memoryWarningPercent") ?: 65.0
                    val samplingMs = call.argument<Int>("samplingIntervalMs")?.toLong() ?: 250L

                    safetyController.configure(
                        MemorySafetyController.SafetyPolicy(
                            maxProcessRssPercent = maxPercent,
                            maxProcessRssMb = maxMb,
                            minAvailableSystemRamMb = minAvailMb,
                            memoryWarningPercent = warnPercent,
                            samplingIntervalMs = samplingMs
                        )
                    )
                    result.success(true)
                }

                "acknowledgeFuseReset" -> {
                    safetyController.acknowledgeAndReset()
                    result.success(true)
                }

                "triggerManualAbort" -> {
                    val reason = call.argument<String>("reason") ?: "User requested manual abort"
                    safetyController.triggerHardAbort(reason)
                    result.success(true)
                }

                "injectSyntheticRss" -> {
                    val rssMb = call.argument<Double?>("rssMb")
                    safetyController.injectSyntheticRss(rssMb)
                    result.success(true)
                }

                "getStorageDirectory" -> {
                    val internalDir = applicationContext.filesDir.absolutePath
                    val externalDir = applicationContext.getExternalFilesDir(null)?.absolutePath ?: internalDir
                    val chk = findCheckpointFile()
                    result.success(mapOf(
                        "internalDir" to internalDir,
                        "externalDir" to externalDir,
                        "checkpointPath" to (chk?.absolutePath ?: ""),
                        "hasCheckpoint" to (chk != null && chk.exists())
                    ))
                }

                "verifyCheckpoint" -> {
                    val file = findCheckpointFile()
                    if (file != null && file.exists()) {
                        result.success(mapOf(
                            "exists" to true,
                            "path" to file.absolutePath,
                            "fileName" to file.name,
                            "sizeMb" to (file.length() / (1024.0 * 1024.0))
                        ))
                    } else {
                        result.success(mapOf(
                            "exists" to false,
                            "path" to (file?.absolutePath ?: "none"),
                            "fileName" to (file?.name ?: "none"),
                            "sizeMb" to 0.0
                        ))
                    }
                }

                "loadModel" -> {
                    thread {
                        try {
                            val chk = findCheckpointFile()
                            if (chk == null || !chk.exists()) {
                                runOnUiThread {
                                    result.error("CHECKPOINT_MISSING", "Checkpoint file not found in storage", null)
                                }
                                return@thread
                            }
                            engine?.close()
                            val e = AndroidSlrEngine(applicationContext, chk, safetyController)
                            engine = e
                            runOnUiThread {
                                result.success(mapOf(
                                    "success" to true,
                                    "checkpointPath" to chk.absolutePath,
                                    "sizeMb" to (chk.length() / (1024.0 * 1024.0)),
                                    "activeParameters" to e.activeParameters,
                                    "activeLayers" to e.activeLayersCount
                                ))
                            }
                        } catch (ex: Exception) {
                            runOnUiThread {
                                result.error("LOAD_FAILED", ex.message, null)
                            }
                        }
                    }
                }

                "executeSingleLayer" -> {
                    thread {
                        try {
                            val activeEngine = engine ?: run {
                                val chk = findCheckpointFile()
                                if (chk != null && chk.exists()) {
                                    val e = AndroidSlrEngine(applicationContext, chk, safetyController)
                                    engine = e
                                    e
                                } else null
                            }

                            if (activeEngine == null) {
                                runOnUiThread {
                                    result.error("CHECKPOINT_MISSING", "Checkpoint file not found in storage", null)
                                }
                                return@thread
                            }

                            val t0 = System.currentTimeMillis()
                            val dummyInput = FloatArray(activeEngine.hiddenSize) { 0.01f }
                            val out = activeEngine.forwardLayer(0, dummyInput, 0)
                            val elapsed = System.currentTimeMillis() - t0

                            val mem = diagnostics.getMemoryMetrics()
                            runOnUiThread {
                                result.success(mapOf(
                                    "status" to "success",
                                    "layerIndex" to 0,
                                    "activeParameters" to activeEngine.activeParameters,
                                    "activeLayers" to activeEngine.activeLayersCount,
                                    "elapsedMs" to elapsed,
                                    "outputDim" to out.size,
                                    "processRssMb" to mem.processRssMb
                                ))
                            }
                        } catch (e: Exception) {
                            runOnUiThread {
                                result.error("EXEC_FAILED", e.message, null)
                            }
                        }
                    }
                }

                "runSafeBenchmark" -> {
                    thread {
                        try {
                            val suite = AndroidBenchmarkSuite(
                                applicationContext,
                                diagnostics,
                                safetyController,
                                findCheckpointFile()
                            )
                            val report = suite.runSafeBenchmark()
                            runOnUiThread {
                                result.success(report.toString())
                            }
                        } catch (e: Exception) {
                            runOnUiThread {
                                result.error("BENCHMARK_ERROR", e.message, null)
                            }
                        }
                    }
                }

                else -> result.notImplemented()
            }
        }
    }

    private fun findCheckpointFile(): File? {
        val candidates = mutableListOf<File>()

        // 1. Internal application storage (preferred & high security)
        candidates.add(File(applicationContext.filesDir, "models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"))
        candidates.add(File(applicationContext.filesDir, "model.safetensors"))
        candidates.add(File(applicationContext.filesDir, "checkpoints/Qwen2.5-1.5B-Instruct/model.safetensors"))

        // 2. External app storage (getExternalFilesDir)
        val extDir = applicationContext.getExternalFilesDir(null)
        if (extDir != null) {
            candidates.add(File(extDir, "models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"))
            candidates.add(File(extDir, "model.safetensors"))
            candidates.add(File(extDir, "checkpoints/Qwen2.5-1.5B-Instruct/model.safetensors"))
        }

        // 3. Absolute standard paths
        candidates.add(File("/data/user/0/com.example.exp07_slr/files/models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"))
        candidates.add(File("/data/user/0/com.example.exp07_slr/files/model.safetensors"))
        candidates.add(File("/sdcard/Android/data/com.example.exp07_slr/files/models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"))
        candidates.add(File("/sdcard/Android/data/com.example.exp07_slr/files/model.safetensors"))
        candidates.add(File("/storage/emulated/0/Android/data/com.example.exp07_slr/files/models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"))
        candidates.add(File("/storage/emulated/0/Android/data/com.example.exp07_slr/files/model.safetensors"))

        return candidates.firstOrNull { it.exists() && it.canRead() } ?: candidates.firstOrNull { it.exists() } ?: candidates[0]
    }

    override fun onDestroy() {
        safetyController.stopMonitoring()
        engine?.close()
        super.onDestroy()
    }
}
