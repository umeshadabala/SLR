package com.example.exp07_slr

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * Dedicated Memory Safety Controller for Selective Layer Residency on Android.
 * Runs independently of Flutter widgets and monitors process RSS and available system memory.
 */
class MemorySafetyController(
    private val context: Context,
    val diagnostics: DeviceDiagnostics
) {

    enum class FuseState {
        NORMAL,
        WARNING,
        ABORT_REQUESTED,
        CLEANUP_IN_PROGRESS,
        ABORTED,
        TERMINATION_REQUESTED,
        TERMINATED,
        CLEANUP_FAILED
    }

    data class SafetyPolicy(
        val maxProcessRssPercent: Double = 75.0,
        val maxProcessRssMb: Double? = null,
        val minAvailableSystemRamMb: Double = 256.0,
        val memoryWarningPercent: Double = 65.0,
        val samplingIntervalMs: Long = 250L,
        val consecutiveBreachSamples: Int = 2,
        val emergencyCleanupTimeoutMs: Long = 3000L,
        val autoTerminateAfterAbort: Boolean = false
    )

    data class StateTransition(
        val fromState: FuseState,
        val toState: FuseState,
        val reason: String,
        val rssMb: Double,
        val availRamMb: Double,
        val activeLayerIdx: Int,
        val timestampMs: Long
    )

    data class FuseEvent(
        val state: FuseState,
        val reason: String,
        val breachMetric: String,
        val breachValue: Double,
        val thresholdValue: Double,
        val timestampMs: Long
    )

    // Current policy and state
    var policy = SafetyPolicy()
        private set

    val currentState = AtomicReference(FuseState.NORMAL)
    val isCancelled = AtomicBoolean(false)

    private val transitions = mutableListOf<StateTransition>()
    private var lastFuseEvent: FuseEvent? = null

    // Tracking active execution context
    var currentActiveLayerIdx = -1
    var currentKvCacheLength = 0
    var currentOperation = "IDLE"

    private var breachSampleCounter = 0
    private var syntheticRssOverrideMb: Double? = null

    private var scheduler: ScheduledExecutorService? = null
    private val listeners = mutableListOf<(FuseState, String) -> Unit>()
    private var cleanupHandler: (() -> Boolean)? = null

    fun addListener(listener: (FuseState, String) -> Unit) {
        synchronized(listeners) {
            listeners.add(listener)
        }
    }

    fun setCleanupHandler(handler: () -> Boolean) {
        cleanupHandler = handler
    }

    fun configure(newPolicy: SafetyPolicy) {
        policy = newPolicy
    }

    /**
     * Injects synthetic RSS measurement for controlled, safe testing of the fuse.
     */
    fun injectSyntheticRss(rssMb: Double?) {
        syntheticRssOverrideMb = rssMb
    }

    fun startMonitoring() {
        stopMonitoring()
        isCancelled.set(false)
        currentState.set(FuseState.NORMAL)
        breachSampleCounter = 0
        scheduler = Executors.newSingleThreadScheduledExecutor { r ->
            val t = Thread(r, "SLR-MemorySafetyMonitor")
            t.isDaemon = true
            t
        }
        scheduler?.scheduleWithFixedDelay(
            { checkMemorySample() },
            0L,
            policy.samplingIntervalMs,
            TimeUnit.MILLISECONDS
        )
    }

    fun stopMonitoring() {
        scheduler?.shutdownNow()
        scheduler = null
    }

    /**
     * Periodic memory check executed on the dedicated background thread.
     */
    private fun checkMemorySample() {
        val state = currentState.get()
        if (state == FuseState.ABORTED || state == FuseState.TERMINATED || state == FuseState.CLEANUP_FAILED) {
            return
        }

        val metrics = diagnostics.getMemoryMetrics()
        val measuredRssMb = syntheticRssOverrideMb ?: metrics.processRssMb
        val availRamMb = metrics.availableSystemRamMb
        val totalRamMb = metrics.totalPhysicalRamMb

        // Effective limits
        val effectiveRssLimitMb = policy.maxProcessRssMb
            ?: (totalRamMb * (policy.maxProcessRssPercent / 100.0))
        val warningRssLimitMb = totalRamMb * (policy.memoryWarningPercent / 100.0)

        // 1. Hard Breach Conditions
        val isRssBreach = effectiveRssLimitMb > 0 && measuredRssMb >= effectiveRssLimitMb
        val isSystemReserveBreach = availRamMb <= policy.minAvailableSystemRamMb

        if (isRssBreach || isSystemReserveBreach) {
            breachSampleCounter++
            if (breachSampleCounter >= policy.consecutiveBreachSamples) {
                val reason = if (isRssBreach) {
                    "Process RSS ($measuredRssMb MB) exceeded limit ($effectiveRssLimitMb MB, ${policy.maxProcessRssPercent}%)"
                } else {
                    "Available system RAM ($availRamMb MB) breached reserve floor (${policy.minAvailableSystemRamMb} MB)"
                }
                val metric = if (isRssBreach) "process_rss_mb" else "available_system_ram_mb"
                val breachVal = if (isRssBreach) measuredRssMb else availRamMb
                val threshVal = if (isRssBreach) effectiveRssLimitMb else policy.minAvailableSystemRamMb

                triggerHardAbort(reason, metric, breachVal, threshVal)
                return
            }
        } else {
            breachSampleCounter = 0
        }

        // 2. Warning Conditions
        val isWarning = measuredRssMb >= warningRssLimitMb || metrics.isLowMemory
        if (isWarning && currentState.get() == FuseState.NORMAL) {
            val warningReason = "Process RSS ($measuredRssMb MB) reached warning threshold ($warningRssLimitMb MB, ${policy.memoryWarningPercent}%)"
            transitionTo(FuseState.WARNING, warningReason, measuredRssMb, availRamMb)
        } else if (!isWarning && currentState.get() == FuseState.WARNING) {
            transitionTo(FuseState.NORMAL, "Memory pressure returned below warning threshold", measuredRssMb, availRamMb)
        }
    }

    /**
     * Atomically triggers hard abort sequence.
     */
    fun triggerHardAbort(
        reason: String,
        breachMetric: String = "manual_abort",
        breachValue: Double = 0.0,
        thresholdValue: Double = 0.0
    ) {
        if (currentState.get() == FuseState.ABORTED || currentState.get() == FuseState.CLEANUP_IN_PROGRESS) {
            return
        }

        isCancelled.set(true)
        lastFuseEvent = FuseEvent(
            state = FuseState.ABORTED,
            reason = reason,
            breachMetric = breachMetric,
            breachValue = breachValue,
            thresholdValue = thresholdValue,
            timestampMs = System.currentTimeMillis()
        )

        val metrics = diagnostics.getMemoryMetrics()
        transitionTo(FuseState.ABORT_REQUESTED, reason, metrics.processRssMb, metrics.availableSystemRamMb)
        executeControlledCleanup(reason)
    }

    /**
     * Safe execution boundary check called before each layer load and token decode.
     * Throws MemorySafetyException if cancelled.
     */
    fun checkBoundary(layerIdx: Int = -1, op: String = "COMPUTE") {
        currentActiveLayerIdx = layerIdx
        currentOperation = op
        if (isCancelled.get()) {
            throw MemorySafetyException(
                "Inference halted by MemorySafetyController: ${lastFuseEvent?.reason ?: "Abort requested"}"
            )
        }
    }

    /**
     * Controlled cleanup sequence executed when hard threshold is reached.
     */
    private fun executeControlledCleanup(reason: String) {
        val metricsPre = diagnostics.getMemoryMetrics()
        transitionTo(FuseState.CLEANUP_IN_PROGRESS, "Executing emergency cleanup", metricsPre.processRssMb, metricsPre.availableSystemRamMb)

        var cleanupSuccess = false
        try {
            cleanupSuccess = cleanupHandler?.invoke() ?: true
        } catch (_: Exception) {
            cleanupSuccess = false
        }

        val metricsPost = diagnostics.getMemoryMetrics()
        val finalState = if (cleanupSuccess) FuseState.ABORTED else FuseState.CLEANUP_FAILED
        transitionTo(finalState, "Cleanup finished (success=$cleanupSuccess)", metricsPost.processRssMb, metricsPost.availableSystemRamMb)

        // Persist crash-resilient diagnostic journal before any potential termination
        persistDiagnosticJournal()

        if (policy.autoTerminateAfterAbort) {
            transitionTo(FuseState.TERMINATION_REQUESTED, "Policy autoTerminateAfterAbort is enabled", metricsPost.processRssMb, metricsPost.availableSystemRamMb)
        }
    }

    private fun transitionTo(newState: FuseState, reason: String, rssMb: Double, availMb: Double) {
        val oldState = currentState.getAndSet(newState)
        if (oldState != newState) {
            val record = StateTransition(
                fromState = oldState,
                toState = newState,
                reason = reason,
                rssMb = rssMb,
                availRamMb = availMb,
                activeLayerIdx = currentActiveLayerIdx,
                timestampMs = System.currentTimeMillis()
            )
            synchronized(transitions) {
                transitions.add(record)
            }
            synchronized(listeners) {
                listeners.forEach { it(newState, reason) }
            }
        }
    }

    /**
     * Atomic persistence of the diagnostic journal to internal app storage.
     */
    fun persistDiagnosticJournal() {
        try {
            val file = File(context.filesDir, "slr_diagnostic_journal.json")
            val root = JSONObject()
            root.put("timestamp_ms", System.currentTimeMillis())
            root.put("current_state", currentState.get().name)
            root.put("is_cancelled", isCancelled.get())
            root.put("policy", JSONObject().apply {
                put("max_process_rss_percent", policy.maxProcessRssPercent)
                put("max_process_rss_mb", policy.maxProcessRssMb ?: JSONObject.NULL)
                put("min_available_system_ram_mb", policy.minAvailableSystemRamMb)
                put("memory_warning_percent", policy.memoryWarningPercent)
            })

            lastFuseEvent?.let { event ->
                root.put("last_fuse_event", JSONObject().apply {
                    put("state", event.state.name)
                    put("reason", event.reason)
                    put("breach_metric", event.breachMetric)
                    put("breach_value", event.breachValue)
                    put("threshold_value", event.thresholdValue)
                    put("timestamp_ms", event.timestampMs)
                })
            }

            val transArray = JSONArray()
            synchronized(transitions) {
                transitions.forEach { t ->
                    transArray.put(JSONObject().apply {
                        put("from_state", t.fromState.name)
                        put("to_state", t.toState.name)
                        put("reason", t.reason)
                        put("rss_mb", t.rssMb)
                        put("avail_ram_mb", t.availRamMb)
                        put("active_layer_idx", t.activeLayerIdx)
                        put("timestamp_ms", t.timestampMs)
                    })
                }
            }
            root.put("transitions", transArray)

            // Atomic file write using temp file and rename
            val tempFile = File(context.filesDir, "slr_diagnostic_journal.tmp")
            FileOutputStream(tempFile).use { fos ->
                fos.write(root.toString(2).toByteArray(Charsets.UTF_8))
                fos.flush()
                fos.fd.sync()
            }
            tempFile.renameTo(file)
        } catch (_: Exception) {
            // Persistence best-effort
        }
    }

    /**
     * Explicit user acknowledgment required to clear an aborted fuse state.
     */
    fun acknowledgeAndReset() {
        if (currentState.get() == FuseState.ABORTED || currentState.get() == FuseState.CLEANUP_FAILED) {
            isCancelled.set(false)
            syntheticRssOverrideMb = null
            breachSampleCounter = 0
            val metrics = diagnostics.getMemoryMetrics()
            transitionTo(FuseState.NORMAL, "Fuse reset acknowledged by user", metrics.processRssMb, metrics.availableSystemRamMb)
        }
    }

    fun getStatusJson(): JSONObject {
        val metrics = diagnostics.getMemoryMetrics()
        val root = JSONObject()
        root.put("state", currentState.get().name)
        root.put("is_cancelled", isCancelled.get())
        root.put("current_rss_mb", syntheticRssOverrideMb ?: metrics.processRssMb)
        root.put("peak_rss_mb", metrics.peakProcessRssMb)
        root.put("available_ram_mb", metrics.availableSystemRamMb)
        root.put("total_ram_mb", metrics.totalPhysicalRamMb)
        root.put("effective_limit_mb", policy.maxProcessRssMb ?: (metrics.totalPhysicalRamMb * (policy.maxProcessRssPercent / 100.0)))
        root.put("current_active_layer_idx", currentActiveLayerIdx)
        root.put("current_kv_cache_length", currentKvCacheLength)
        root.put("last_event", lastFuseEvent?.reason ?: "")
        return root
    }
}

class MemorySafetyException(message: String) : RuntimeException(message)
