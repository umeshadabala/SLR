package com.example.exp07_slr

import android.content.Intent
import android.os.Bundle
import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread

class MainActivity : FlutterActivity() {

    private val CHANNEL = "com.example.exp07_slr/channel"

    private lateinit var methodChannel: MethodChannel
    private val diagnostics: DeviceDiagnostics by lazy { DeviceDiagnostics(applicationContext) }
    private val safetyController: MemorySafetyController by lazy { MemorySafetyController(applicationContext, diagnostics) }
    private var engine: AndroidSlrEngine? = null
    private var tokenizer: QwenTokenizer? = null

    private val isBenchmarkRunning = java.util.concurrent.atomic.AtomicBoolean(false)
    private val isGenerationRunning = java.util.concurrent.atomic.AtomicBoolean(false)
    private var activeBenchmarkSuite: AndroidBenchmarkSuite? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        handleIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        val action = intent?.getStringExtra("action") ?: return
        when (action) {
            "run_benchmark" -> {
                if (!isBenchmarkRunning.compareAndSet(false, true)) {
                    Log.w("SLR_NATIVE", "Benchmark already running, ignoring duplicate intent trigger.")
                    return
                }
                thread {
                    try {
                        Log.i("SLR_NATIVE", "Triggering Automated 18-Test Safe Benchmark from intent...")
                        initializeEngineIfAvailable()
                        val suite = AndroidBenchmarkSuite(
                            applicationContext,
                            diagnostics,
                            safetyController,
                            findCheckpointFile(),
                            engine,
                            tokenizer
                        )
                        activeBenchmarkSuite = suite
                        val report = suite.runSafeBenchmark()
                        val runsDir = File(applicationContext.filesDir, "runs")
                        if (!runsDir.exists()) runsDir.mkdirs()
                        val benchFile = File(runsDir, "benchmark_latest.json")
                        benchFile.writeText(report.toString(2))
                        Log.i("SLR_NATIVE", "Benchmark finished successfully! Saved to ${benchFile.absolutePath}")
                    } catch (e: Exception) {
                        Log.e("SLR_NATIVE", "Benchmark failed: ${e.message}", e)
                    } finally {
                        activeBenchmarkSuite = null
                        isBenchmarkRunning.set(false)
                    }
                }
            }
            "test_generation" -> {
                if (!isGenerationRunning.compareAndSet(false, true)) {
                    Log.w("SLR_NATIVE", "Generation already running, ignoring duplicate intent trigger.")
                    return
                }
                val prompt = intent.getStringExtra("prompt") ?: "Explain Selective Layer Residency."
                val maxTokens = intent.getIntExtra("max_tokens", 8)
                thread {
                    try {
                        Log.i("SLR_NATIVE", "Triggering test generation for: '$prompt' (maxTokens=$maxTokens)")
                        initializeEngineIfAvailable()
                        val eng = engine
                        if (eng == null) {
                            Log.e("SLR_NATIVE", "Engine initialization failed! Engine is null.")
                            return@thread
                        }
                        eng.generateStream(
                            prompt = prompt,
                            messages = null,
                            maxNewTokens = maxTokens,
                            temperature = 0.0,
                            topP = 1.0,
                            onToken = { tokenId, text, elapsedSec, isFirst ->
                                Log.i("SLR_NATIVE", "[TOKEN] id=$tokenId text='$text' elapsed=${elapsedSec}s isFirst=$isFirst")
                            },
                            onDone = { report ->
                                Log.i("SLR_NATIVE", "[DONE] Generation completed! Report:\n${report.toString(2)}")
                            },
                            onError = { err ->
                                Log.e("SLR_NATIVE", "[ERROR] Generation error: $err")
                            }
                        )
                    } catch (e: Exception) {
                        Log.e("SLR_NATIVE", "Generation failed: ${e.message}", e)
                    } finally {
                        isGenerationRunning.set(false)
                    }
                }
            }
        }
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        safetyController.startMonitoring()
        methodChannel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
        initializeEngineIfAvailable()

        methodChannel.setMethodCallHandler { call, result ->
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
                        "rssAnonMb" to m.rssAnonMb,
                        "rssFileMb" to m.rssFileMb,
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
                    val vFile = findVocabFile()
                    val exists = file != null && file.exists() && file.length() > 100 * 1024 * 1024
                    result.success(mapOf(
                        "exists" to exists,
                        "path" to (file?.absolutePath ?: "none"),
                        "fileName" to (file?.name ?: "none"),
                        "sizeMb" to (if (exists) file!!.length() / (1024.0 * 1024.0) else 0.0),
                        "hasTokenizer" to (vFile != null && vFile.exists()),
                        "tokenizerPath" to (vFile?.absolutePath ?: "")
                    ))
                }

                "loadModel" -> {
                    thread {
                        try {
                            val ok = initializeEngineIfAvailable()
                            val chk = findCheckpointFile()
                            val e = engine
                            if (ok && chk != null && e != null) {
                                runOnUiThread {
                                    result.success(mapOf(
                                        "success" to true,
                                        "checkpointPath" to chk.absolutePath,
                                        "sizeMb" to (chk.length() / (1024.0 * 1024.0)),
                                        "activeParameters" to e.activeParameters,
                                        "activeLayers" to e.activeLayersCount
                                    ))
                                }
                            } else {
                                runOnUiThread {
                                    result.error("LOAD_FAILED", "Failed to load model or tokenizer from storage", null)
                                }
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
                                initializeEngineIfAvailable()
                                engine
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
                    if (!isBenchmarkRunning.compareAndSet(false, true)) {
                        result.error("BENCHMARK_BUSY", "Benchmark is already running", null)
                        return@setMethodCallHandler
                    }
                    thread {
                        try {
                            val suite = AndroidBenchmarkSuite(
                                applicationContext,
                                diagnostics,
                                safetyController,
                                findCheckpointFile(),
                                engine,
                                tokenizer
                            )
                            activeBenchmarkSuite = suite
                            val report = suite.runSafeBenchmark()
                            runOnUiThread {
                                result.success(report.toString())
                            }
                        } catch (e: Exception) {
                            runOnUiThread {
                                result.error("BENCHMARK_ERROR", e.message, null)
                            }
                        } finally {
                            activeBenchmarkSuite = null
                            isBenchmarkRunning.set(false)
                        }
                    }
                }

                "cancelBenchmark" -> {
                    val suite = activeBenchmarkSuite
                    if (suite != null) {
                        suite.cancel()
                        result.success(true)
                    } else {
                        result.success(false)
                    }
                }

                "startNativeGeneration" -> {
                    val prompt = call.argument<String>("prompt") ?: ""
                    val messagesRaw = call.argument<List<Map<String, String>>>("messages")
                    val maxNewTokens = call.argument<Int>("maxNewTokens") ?: 32
                    val temperature = call.argument<Double>("temperature") ?: 0.0
                    val topP = call.argument<Double>("topP") ?: 1.0

                    thread {
                        try {
                            if (engine == null || tokenizer == null) {
                                initializeEngineIfAvailable()
                            }
                            val eng = engine
                            if (eng == null) {
                                runOnUiThread {
                                    methodChannel.invokeMethod("onNativeError", mapOf(
                                        "error" to "Android SLR Engine could not be initialized from local storage"
                                    ))
                                }
                                return@thread
                            }

                            eng.generateStream(
                                prompt = prompt,
                                messages = messagesRaw,
                                maxNewTokens = maxNewTokens,
                                temperature = temperature,
                                topP = topP,
                                onToken = { tokenId, text, elapsedSec, isFirst ->
                                    runOnUiThread {
                                        methodChannel.invokeMethod("onNativeToken", mapOf(
                                            "tokenId" to tokenId,
                                            "text" to text,
                                            "elapsedSec" to elapsedSec,
                                            "isFirst" to isFirst
                                        ))
                                    }
                                },
                                onDone = { reportJson ->
                                    runOnUiThread {
                                        methodChannel.invokeMethod("onNativeDone", mapOf(
                                            "report" to reportJson.toString()
                                        ))
                                    }
                                },
                                onError = { errMsg ->
                                    runOnUiThread {
                                        methodChannel.invokeMethod("onNativeError", mapOf(
                                            "error" to errMsg
                                        ))
                                    }
                                }
                            )
                        } catch (e: Exception) {
                            runOnUiThread {
                                methodChannel.invokeMethod("onNativeError", mapOf(
                                    "error" to (e.message ?: "Native generation failed")
                                ))
                            }
                        }
                    }
                    result.success(true)
                }

                "cancelNativeGeneration" -> {
                    engine?.cancel()
                    result.success(true)
                }

                "listNativeRuns" -> {
                    try {
                        val runsDir = File(applicationContext.filesDir, "runs")
                        val reports = mutableListOf<Map<String, Any?>>()
                        if (runsDir.exists()) {
                            val files = runsDir.listFiles { f -> f.extension == "json" }
                                ?.sortedByDescending { it.lastModified() }
                                ?: emptyList()
                            for (f in files.take(50)) {
                                try {
                                    val obj = JSONObject(f.readText())
                                    val map = jsonObjectToMap(obj)
                                    reports.add(map)
                                } catch (_: Exception) {}
                            }
                        }
                        result.success(reports)
                    } catch (e: Exception) {
                        result.error("LIST_FAILED", e.message, null)
                    }
                }

                else -> result.notImplemented()
            }
        }
    }

    private fun jsonObjectToMap(obj: JSONObject): Map<String, Any?> {
        val map = mutableMapOf<String, Any?>()
        val keys = obj.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            val v = obj.get(k)
            map[k] = when (v) {
                is JSONObject -> jsonObjectToMap(v)
                is JSONArray -> jsonArrayToList(v)
                JSONObject.NULL -> null
                else -> v
            }
        }
        return map
    }

    private fun jsonArrayToList(arr: JSONArray): List<Any?> {
        val list = mutableListOf<Any?>()
        for (i in 0 until arr.length()) {
            val v = arr.get(i)
            list.add(when (v) {
                is JSONObject -> jsonObjectToMap(v)
                is JSONArray -> jsonArrayToList(v)
                JSONObject.NULL -> null
                else -> v
            })
        }
        return list
    }

    private fun initializeEngineIfAvailable(): Boolean {
        val chk = findCheckpointFile() ?: return false
        val vocab = findVocabFile() ?: return false
        val merges = findMergesFile()

        try {
            if (tokenizer == null) {
                tokenizer = QwenTokenizer(vocab, merges)
            }
            if (engine == null) {
                engine = AndroidSlrEngine(applicationContext, chk, safetyController, tokenizer)
            }
            return true
        } catch (e: Exception) {
            e.printStackTrace()
            return false
        }
    }

    private fun findCheckpointFile(): File? {
        val candidates = mutableListOf<File>()
        candidates.add(File(applicationContext.filesDir, "models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"))
        candidates.add(File(applicationContext.filesDir, "model.safetensors"))
        candidates.add(File(applicationContext.filesDir, "checkpoints/Qwen2.5-1.5B-Instruct/model.safetensors"))

        val extDir = applicationContext.getExternalFilesDir(null)
        if (extDir != null) {
            candidates.add(File(extDir, "models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"))
            candidates.add(File(extDir, "model.safetensors"))
            candidates.add(File(extDir, "checkpoints/Qwen2.5-1.5B-Instruct/model.safetensors"))
        }

        candidates.add(File("/data/user/0/com.example.exp07_slr/files/models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"))
        candidates.add(File("/data/user/0/com.example.exp07_slr/files/model.safetensors"))
        candidates.add(File("/sdcard/Android/data/com.example.exp07_slr/files/models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"))

        return candidates.firstOrNull { it.exists() && it.canRead() } ?: candidates.firstOrNull { it.exists() }
    }

    private fun findVocabFile(): File? {
        val candidates = mutableListOf<File>()
        candidates.add(File(applicationContext.filesDir, "models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/vocab.json"))
        candidates.add(File(applicationContext.filesDir, "vocab.json"))
        candidates.add(File(applicationContext.filesDir, "checkpoints/Qwen2.5-1.5B-Instruct/vocab.json"))

        val extDir = applicationContext.getExternalFilesDir(null)
        if (extDir != null) {
            candidates.add(File(extDir, "models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/vocab.json"))
            candidates.add(File(extDir, "vocab.json"))
        }
        candidates.add(File("/data/user/0/com.example.exp07_slr/files/models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/vocab.json"))
        candidates.add(File("/sdcard/Android/data/com.example.exp07_slr/files/models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/vocab.json"))

        return candidates.firstOrNull { it.exists() && it.canRead() } ?: candidates.firstOrNull { it.exists() }
    }

    private fun findMergesFile(): File? {
        val candidates = mutableListOf<File>()
        candidates.add(File(applicationContext.filesDir, "models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/merges.txt"))
        candidates.add(File(applicationContext.filesDir, "merges.txt"))
        candidates.add(File(applicationContext.filesDir, "checkpoints/Qwen2.5-1.5B-Instruct/merges.txt"))

        val extDir = applicationContext.getExternalFilesDir(null)
        if (extDir != null) {
            candidates.add(File(extDir, "models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/merges.txt"))
            candidates.add(File(extDir, "merges.txt"))
        }
        candidates.add(File("/data/user/0/com.example.exp07_slr/files/models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/merges.txt"))
        candidates.add(File("/sdcard/Android/data/com.example.exp07_slr/files/models/qwen2.5-1.5b-instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/merges.txt"))

        return candidates.firstOrNull { it.exists() && it.canRead() } ?: candidates.firstOrNull { it.exists() }
    }

    override fun onDestroy() {
        safetyController.stopMonitoring()
        engine?.close()
        super.onDestroy()
    }
}
