package com.example.exp07_slr

import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Pure binary reader for Hugging Face Safetensors format on Android.
 * Enables zero-overhead random-access slicing of individual transformer layer weights.
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

    /**
     * Reads a tensor directly from disk and converts BF16/FP32 to a FloatArray using memory mapping.
     */
    fun readTensorAsFloatArray(name: String): FloatArray {
        val meta = tensorMap[name] ?: throw NoSuchElementException("Tensor not found: $name")
        val numElements = meta.numElements.toInt()
        val result = FloatArray(numElements)

        val mappedBuffer = channel.map(FileChannel.MapMode.READ_ONLY, meta.dataStartByte, meta.dataLengthBytes)
        mappedBuffer.order(ByteOrder.LITTLE_ENDIAN)

        when (meta.dtype.uppercase()) {
            "BF16", "BFLOAT16" -> {
                val shortBuffer = mappedBuffer.asShortBuffer()
                for (i in 0 until numElements) {
                    val rawShort = shortBuffer.get()
                    // BF16 is identical to the top 16 bits of IEEE 754 Float32
                    val intBits = (rawShort.toInt() and 0xFFFF) shl 16
                    result[i] = java.lang.Float.intBitsToFloat(intBits)
                }
            }
            "F32", "FLOAT32" -> {
                val floatBuffer = mappedBuffer.asFloatBuffer()
                floatBuffer.get(result)
            }
            else -> throw UnsupportedOperationException("Unsupported tensor dtype: ${meta.dtype}")
        }

        return result
    }

    override fun close() {
        try {
            channel.close()
            raf.close()
        } catch (_: Exception) {}
    }
}
