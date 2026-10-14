package com.example.exp07_slr

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.Process
import java.io.File
import java.io.RandomAccessFile

/**
 * Native Android diagnostics component.
 * Retrieves genuine device specifications and operating system process memory measurements
 * directly from Android system APIs and /proc/self/status.
 */
class DeviceDiagnostics(private val context: Context) {

    data class MemoryMetrics(
        val processRssMb: Double,
        val processRssSource: String,
        val peakProcessRssMb: Double,
        val processPssMb: Double,
        val availableSystemRamMb: Double,
        val totalPhysicalRamMb: Double,
        val isLowMemory: Boolean,
        val systemMemoryThresholdMb: Double,
        val timestampMs: Long
    )

    data class DeviceInfo(
        val manufacturer: String,
        val model: String,
        val androidVersion: String,
        val apiLevel: Int,
        val supportedAbis: List<String>,
        val cpuCoreCount: Int,
        val totalPhysicalRamMb: Double,
        val appVersion: String,
        val nativeRuntimeVersion: String
    )

    fun getDeviceInfo(): DeviceInfo {
        val totalRamMb = getSystemMemoryInfo().first
        val pInfo = try {
            context.packageManager.getPackageInfo(context.packageName, 0)
        } catch (e: Exception) {
            null
        }
        val appVer = pInfo?.versionName ?: "1.0.0"

        return DeviceInfo(
            manufacturer = Build.MANUFACTURER ?: "Unknown",
            model = Build.MODEL ?: "Unknown",
            androidVersion = Build.VERSION.RELEASE ?: "Unknown",
            apiLevel = Build.VERSION.SDK_INT,
            supportedAbis = Build.SUPPORTED_ABIS?.toList() ?: listOf("unknown"),
            cpuCoreCount = Runtime.getRuntime().availableProcessors(),
            totalPhysicalRamMb = totalRamMb,
            appVersion = appVer,
            nativeRuntimeVersion = "Android ART / Linux kernel " + System.getProperty("os.version", "unknown")
        )
    }

    /**
     * Reads resident set size (VmRSS) directly from /proc/self/status.
     * Returns pair of (VmRSS_MB, VmHWM_MB).
     */
    fun readProcStatusMemory(): Pair<Double, Double> {
        var vmRssKb = -1.0
        var vmHwmKb = -1.0
        try {
            val statusFile = File("/proc/self/status")
            if (statusFile.exists() && statusFile.canRead()) {
                statusFile.forEachLine { line ->
                    if (line.startsWith("VmRSS:")) {
                        val parts = line.split("\\s+".toRegex())
                        if (parts.size >= 2) {
                            vmRssKb = parts[1].toDoubleOrNull() ?: -1.0
                        }
                    } else if (line.startsWith("VmHWM:")) {
                        val parts = line.split("\\s+".toRegex())
                        if (parts.size >= 2) {
                            vmHwmKb = parts[1].toDoubleOrNull() ?: -1.0
                        }
                    }
                }
            }
        } catch (_: Exception) {
            // Fallback handled below
        }

        // If /proc/self/status is restricted by SELinux, fallback to Debug.MemoryInfo
        if (vmRssKb < 0) {
            val memInfo = Debug.MemoryInfo()
            Debug.getMemoryInfo(memInfo)
            val totalPss = memInfo.totalPss.toDouble() // in kB
            vmRssKb = totalPss
            vmHwmKb = totalPss
        }

        val rssMb = if (vmRssKb >= 0) vmRssKb / 1024.0 else 0.0
        val hwmMb = if (vmHwmKb >= 0) vmHwmKb / 1024.0 else rssMb
        return Pair(rssMb, hwmMb)
    }

    /**
     * Retrieves system RAM from ActivityManager.MemoryInfo.
     * Returns pair of (totalRamMb, availRamMb, isLowMem, thresholdMb).
     */
    private fun getSystemMemoryInfo(): Triple<Double, Double, Boolean> {
        val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        if (actManager != null) {
            actManager.getMemoryInfo(memInfo)
            val totalMb = memInfo.totalMem.toDouble() / (1024.0 * 1024.0)
            val availMb = memInfo.availMem.toDouble() / (1024.0 * 1024.0)
            return Triple(totalMb, availMb, memInfo.lowMemory)
        }
        return Triple(0.0, 0.0, false)
    }

    fun getMemoryMetrics(): MemoryMetrics {
        val (procRssMb, procHwmMb) = readProcStatusMemory()
        val (totalRamMb, availRamMb, isLow) = getSystemMemoryInfo()

        var pssMb = 0.0
        try {
            val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            val pids = intArrayOf(Process.myPid())
            val memInfos = actManager?.getProcessMemoryInfo(pids)
            if (!memInfos.isNullOrEmpty()) {
                pssMb = memInfos[0].totalPss.toDouble() / 1024.0
            }
        } catch (_: Exception) {
            pssMb = procRssMb
        }

        return MemoryMetrics(
            processRssMb = procRssMb,
            processRssSource = if (File("/proc/self/status").canRead()) "/proc/self/status (VmRSS)" else "Debug.MemoryInfo (TotalPss)",
            peakProcessRssMb = procHwmMb,
            processPssMb = pssMb,
            availableSystemRamMb = availRamMb,
            totalPhysicalRamMb = totalRamMb,
            isLowMemory = isLow,
            systemMemoryThresholdMb = totalRamMb * 0.1, // Approximate system low threshold
            timestampMs = System.currentTimeMillis()
        )
    }
}
