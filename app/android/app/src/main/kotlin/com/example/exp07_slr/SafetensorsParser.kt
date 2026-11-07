package com.example.exp07_slr

import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import kotlin.math.min

/**
 * High-performance binary reader for Hugging Face Safetensors format on Android.
 * Features zero-overhead random-access slicing, parallel vectorized BF16-to-FP32 unpacking,
 * strict layer-boundary buffer eviction for true 1/28 Selective Layer Residency,
 * and direct writing into pre-allocated reusable layer buffers.
 */
class SafetensorsParser(private val file: File) : AutoCloseable {

    data class TensorMeta(
        val name: String,
        val dtype: String,
        val shape: LongArray,
        val dataStartByte: Long,
        val dataLengthBytes: Long
    ) {
        val numElements: Long
            get() = shape.fold(1L) { acc, dim -> acc * dim }
    }

    private val raf: RandomAccessFile = RandomAccessFile(file, "r")
    private val channel: FileChannel = raf.channel
    private val tensorMap = mutableMapOf<String, TensorMeta>()

    // Persistent cache only for tensors shared across all layers (e.g. embed_tokens, norm)
    private val persistentBufferCache = mutableMapOf<String, ByteBuffer>()

    // Temporary cache for the currently active layer; strictly evicted on layer release
    private val activeLayerBufferCache = mutableMapOf<String, ByteBuffer>()

    var headerLengthBytes: Long = 0
        private set

    init {
        readHeader()
    }

    private fun readHeader() {
        val lenBuffer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
        channel.read(lenBuffer, 0L)
        lenBuffer.flip()
        headerLengthBytes = lenBuffer.long

        if (headerLengthBytes <= 0 || headerLengthBytes > 50 * 1024 * 1024) {
            throw IllegalArgumentException("Invalid Safetensors header length: $headerLengthBytes")
        }

        val headerBuffer = ByteBuffer.allocate(headerLengthBytes.toInt())
        channel.read(headerBuffer, 8L)
        headerBuffer.flip()

        val jsonString = Charsets.UTF_8.decode(headerBuffer).toString()
        val root = JSONObject(jsonString)

        val baseDataOffset = 8L + headerLengthBytes

        val keys = root.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            if (key == "__metadata__") continue

            val tensorJson = root.getJSONObject(key)
            val dtype = tensorJson.getString("dtype")
            val shapeArray = tensorJson.getJSONArray("shape")
            val shape = LongArray(shapeArray.length()) { i -> shapeArray.getLong(i) }
            val offsetsArray = tensorJson.getJSONArray("data_offsets")
            val offsetStart = offsetsArray.getLong(0)
            val offsetEnd = offsetsArray.getLong(1)

            val meta = TensorMeta(
                name = key,
                dtype = dtype,
                shape = shape,
                dataStartByte = baseDataOffset + offsetStart,
                dataLengthBytes = offsetEnd - offsetStart
            )
            tensorMap[key] = meta
        }
    }

    fun hasTensor(name: String): Boolean = tensorMap.containsKey(name)

    fun getTensorMeta(name: String): TensorMeta? = tensorMap[name]

    fun getAllTensorNames(): List<String> = tensorMap.keys.toList()

    private fun isPersistentTensor(name: String): Boolean {
        return name == "model.embed_tokens.weight" ||
               name == "model.norm.weight" ||
               name == "lm_head.weight"
    }

    /**
     * Returns a memory-mapped ByteBuffer for the specified tensor.
     * Persistent model-wide tensors are retained; layer tensors are isolated to the active layer cache.
     */
    fun getMappedBuffer(name: String): ByteBuffer {
        if (isPersistentTensor(name)) {
            val cached = persistentBufferCache[name]
            if (cached != null) return cached.duplicate().order(ByteOrder.LITTLE_ENDIAN)
            val meta = tensorMap[name] ?: throw NoSuchElementException("Tensor not found: $name")
            val buf = channel.map(FileChannel.MapMode.READ_ONLY, meta.dataStartByte, meta.dataLengthBytes)
            buf.order(ByteOrder.LITTLE_ENDIAN)
            persistentBufferCache[name] = buf
            return buf.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        }

        val cached = activeLayerBufferCache[name]
        if (cached != null) return cached.duplicate().order(ByteOrder.LITTLE_ENDIAN)

        val meta = tensorMap[name] ?: throw NoSuchElementException("Tensor not found: $name")
        val buf = channel.map(FileChannel.MapMode.READ_ONLY, meta.dataStartByte, meta.dataLengthBytes)
        buf.order(ByteOrder.LITTLE_ENDIAN)
        activeLayerBufferCache[name] = buf
        return buf.duplicate().order(ByteOrder.LITTLE_ENDIAN)
    }

    var enableMadviseEviction: Boolean = true

    /**
     * Evicts mapped buffers and memory pages for the released layer to preserve the strict 1/28 Selective Layer Residency invariant.
     * When enableMadviseEviction is false, clean file-backed pages remain in OS page cache avoiding redundant flash I/O,
     * while active layer buffer cache is strictly cleared to maintain single-layer residency in memory.
     */
    fun evictLayerTensors(layerIdx: Int, forceEvictPages: Boolean = false) {
        activeLayerBufferCache.clear()
        if (AndroidSlrEngine.hasNativeLib && (enableMadviseEviction || forceEvictPages)) {
            val prefix = "model.layers.$layerIdx."
            var minStart = Long.MAX_VALUE
            var maxEnd = 0L
            for ((tName, meta) in tensorMap) {
                if (tName.startsWith(prefix)) {
                    if (meta.dataStartByte < minStart) minStart = meta.dataStartByte
                    val end = meta.dataStartByte + meta.dataLengthBytes
                    if (end > maxEnd) maxEnd = end
                }
            }
            if (minStart < maxEnd) {
                AndroidSlrEngine.nativeEvictRange(minStart, maxEnd - minStart)
            }
        }
    }

    /**
     * High-speed zero-allocation tensor read directly into a pre-allocated FloatArray target.
     * Uses ARM NEON native kernel directly from mmap pointer when available, or optimized bulk primitive unpacking fallback.
     */
    fun readTensorIntoFloatArray(
        name: String,
        target: FloatArray,
        executor: ExecutorService? = null,
        numWorkers: Int = 8
    ) {
        val meta = tensorMap[name] ?: throw NoSuchElementException("Tensor not found: $name")
        val numElements = meta.numElements.toInt()

        if (AndroidSlrEngine.hasNativeLib) {
            AndroidSlrEngine.nativeUnpackBf16FromOffset(meta.dataStartByte, target, numElements)
            return
        }

        val mappedBuffer = getMappedBuffer(name)

        when (meta.dtype.uppercase()) {
            "BF16", "BFLOAT16" -> {
                val numPairs = numElements / 2
                val chunkSize = 65536
                val numChunks = (numPairs + chunkSize - 1) / chunkSize

                if (executor != null && numWorkers > 1 && numChunks > 1) {
                    val latch = CountDownLatch(numWorkers)
                    val workerChunks = (numChunks + numWorkers - 1) / numWorkers

                    for (w in 0 until numWorkers) {
                        val startC = w * workerChunks
                        val endC = min(startC + workerChunks, numChunks)
                        if (startC >= numChunks) {
                            latch.countDown()
                            continue
                        }
                        executor.execute {
                            try {
                                val threadBuf = mappedBuffer.duplicate().order(ByteOrder.LITTLE_ENDIAN).asIntBuffer()
                                val localInts = IntArray(chunkSize)
                                for (c in startC until endC) {
                                    val startP = c * chunkSize
                                    val countP = min(chunkSize, numPairs - startP)
                                    threadBuf.position(startP)
                                    threadBuf.get(localInts, 0, countP)

                                    var tIdx = startP * 2
                                    for (i in 0 until countP) {
                                        val pair = localInts[i]
                                        target[tIdx] = java.lang.Float.intBitsToFloat((pair and 0xFFFF) shl 16)
                                        target[tIdx + 1] = java.lang.Float.intBitsToFloat(pair and 0xFFFF0000.toInt())
                                        tIdx += 2
                                    }
                                }
                            } finally {
                                latch.countDown()
                            }
                        }
                    }
                    latch.await()
                } else {
                    val threadBuf = mappedBuffer.asIntBuffer()
                    val localInts = IntArray(min(numPairs, chunkSize))
                    var startP = 0
                    while (startP < numPairs) {
                        val countP = min(localInts.size, numPairs - startP)
                        threadBuf.get(localInts, 0, countP)
                        var tIdx = startP * 2
                        for (i in 0 until countP) {
                            val pair = localInts[i]
                            target[tIdx] = java.lang.Float.intBitsToFloat((pair and 0xFFFF) shl 16)
                            target[tIdx + 1] = java.lang.Float.intBitsToFloat(pair and 0xFFFF0000.toInt())
                            tIdx += 2
                        }
                        startP += countP
                    }
                }

                if (numElements % 2 != 0) {
                    val lastShort = mappedBuffer.getShort((numElements - 1) * 2)
                    target[numElements - 1] = java.lang.Float.intBitsToFloat((lastShort.toInt() and 0xFFFF) shl 16)
                }
            }
            "F32", "FLOAT32" -> {
                val floatBuffer = mappedBuffer.asFloatBuffer()
                floatBuffer.get(target, 0, numElements)
            }
            else -> throw UnsupportedOperationException("Unsupported tensor dtype: ${meta.dtype}")
        }
    }

    /**
     * Convenience legacy method: allocates a new array if reusable target is not supplied.
     */
    fun readTensorAsFloatArray(name: String): FloatArray {
        val meta = tensorMap[name] ?: throw NoSuchElementException("Tensor not found: $name")
        val numElements = meta.numElements.toInt()
        val result = FloatArray(numElements)
        readTensorIntoFloatArray(name, result)
        return result
    }

    override fun close() {
        try {
            persistentBufferCache.clear()
            activeLayerBufferCache.clear()
            channel.close()
            raf.close()
        } catch (_: Exception) {}
    }
}
