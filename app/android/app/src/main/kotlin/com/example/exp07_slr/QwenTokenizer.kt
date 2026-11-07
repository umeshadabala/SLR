package com.example.exp07_slr

import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets

/**
 * Exact Byte-Level BPE Tokenizer for Qwen2.5 on Android.
 * Implements byte-to-unicode mapping, BPE merge rules, and ChatML templating.
 */
class QwenTokenizer(
    private val vocabFile: File,
    private val mergesFile: File? = null
) {
    companion object {
        const val IM_START_ID = 151644
        const val IM_END_ID = 151645
        const val EOS_ID = 151645
        const val ENDOFTEXT_ID = 151643
        const val IM_START_STR = "<|im_start|>"
        const val IM_END_STR = "<|im_end|>"
        const val ENDOFTEXT_STR = "<|endoftext|>"
    }

    private val vocab = HashMap<String, Int>(160000)
    private val invVocab = HashMap<Int, String>(160000)
    private val bpeRanks = HashMap<Long, Int>(160000) // key = (first.hashCode() shl 32) or second.hashCode() or String pair

    // Merges string pair map
    private val mergesMap = HashMap<String, Int>(160000)

    // Byte to Unicode mapping table
    private val byteToUnicode = IntArray(256)
    private val unicodeToByte = HashMap<Char, Int>(256)

    init {
        initByteToUnicode()
        loadVocab()
        if (mergesFile != null && mergesFile.exists()) {
            loadMerges()
        }
    }

    private fun initByteToUnicode() {
        val bs = mutableListOf<Int>()
        // '!' to '~' (33..126)
        for (i in 33..126) bs.add(i)
        // '¡' to '¬' (161..172)
        for (i in 161..172) bs.add(i)
        // '®' to 'ÿ' (174..255)
        for (i in 174..255) bs.add(i)

        val cs = ArrayList(bs)
        var n = 0
        for (b in 0..255) {
            if (!bs.contains(b)) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }

        for (i in bs.indices) {
            val b = bs[i]
            val c = cs[i].toChar()
            byteToUnicode[b] = c.code
            unicodeToByte[c] = b
        }
    }

    private fun loadVocab() {
        val text = vocabFile.readText(StandardCharsets.UTF_8)
        val json = JSONObject(text)
        val keys = json.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            val id = json.getInt(k)
            vocab[k] = id
            invVocab[id] = k
        }
        // Explicitly guarantee special tokens
        vocab[IM_START_STR] = IM_START_ID
        vocab[IM_END_STR] = IM_END_ID
        vocab[ENDOFTEXT_STR] = ENDOFTEXT_ID
        invVocab[IM_START_ID] = IM_START_STR
        invVocab[IM_END_ID] = IM_END_STR
        invVocab[ENDOFTEXT_ID] = ENDOFTEXT_STR
    }

    private fun loadMerges() {
        var rank = 0
        mergesFile?.forEachLine(StandardCharsets.UTF_8) { line ->
            val trimmed = line.trim()
            if (trimmed.isNotEmpty() && !trimmed.startsWith("#")) {
                val parts = trimmed.split(" ")
                if (parts.size >= 2) {
                    val key = "${parts[0]} ${parts[1]}"
                    mergesMap[key] = rank++
                }
            }
        }
    }

    fun hasVocab(): Boolean = vocab.isNotEmpty()

    /**
     * Converts a single byte into its BPE unicode character.
     */
    private fun byteToChar(b: Int): Char {
        val u = b and 0xFF
        return byteToUnicode[u].toChar()
    }

    /**
     * BPE merge algorithm for a sequence of unicode characters.
     */
    private fun bpe(token: String): List<String> {
        var word = ArrayList<String>(token.length)
        for (i in 0 until token.length) {
            word.add(token[i].toString())
        }
        if (word.size <= 1) return word

        while (true) {
            var minRank = Int.MAX_VALUE
            var bestFirst = ""
            var bestSecond = ""
            var bestIdx = -1

            for (i in 0 until word.size - 1) {
                val pairKey = "${word[i]} ${word[i + 1]}"
                val r = mergesMap[pairKey]
                if (r != null && r < minRank) {
                    minRank = r
                    bestFirst = word[i]
                    bestSecond = word[i + 1]
                    bestIdx = i
                }
            }

            if (minRank == Int.MAX_VALUE) break

            val newWord = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == bestFirst && word[i + 1] == bestSecond) {
                    newWord.add(bestFirst + bestSecond)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i++
                }
            }
            word = newWord
            if (word.size == 1) break
        }
        return word
    }

    /**
     * Encodes a plain text chunk into token IDs using BPE.
     */
    fun encodeChunk(text: String): List<Int> {
        val result = mutableListOf<Int>()
        // Regex splitting following GPT-4/Qwen pattern
        val regex = java.util.regex.Pattern.compile(
            "'(?:[sStT]|re|ve|m|ll|d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{N}\\p{M}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        )
        val matcher = regex.matcher(text)
        var lastEnd = 0

        while (matcher.find()) {
            val word = matcher.group()
            val utf8Bytes = word.toByteArray(StandardCharsets.UTF_8)
            val sb = StringBuilder(utf8Bytes.size)
            for (b in utf8Bytes) {
                sb.append(byteToChar(b.toInt()))
            }
            val bpeStr = sb.toString()
            val pieces = bpe(bpeStr)
            for (p in pieces) {
                val id = vocab[p]
                if (id != null) {
                    result.add(id)
                } else {
                    // Fallback to byte tokens
                    for (ch in p) {
                        val singleId = vocab[ch.toString()]
                        if (singleId != null) result.add(singleId)
                    }
                }
            }
            lastEnd = matcher.end()
        }

        // Catch any remaining tail characters
        if (lastEnd < text.length) {
            val tail = text.substring(lastEnd)
            val utf8Bytes = tail.toByteArray(StandardCharsets.UTF_8)
            for (b in utf8Bytes) {
                val ch = byteToChar(b.toInt()).toString()
                val id = vocab[ch]
                if (id != null) result.add(id)
            }
        }

        return result
    }

    /**
     * Encodes arbitrary text into Qwen token IDs, recognizing ChatML special tokens.
     */
    fun encode(text: String): List<Int> {
        val result = mutableListOf<Int>()
        var remaining = text

        val specialTokens = listOf(
            IM_START_STR to IM_START_ID,
            IM_END_STR to IM_END_ID,
            ENDOFTEXT_STR to ENDOFTEXT_ID
        )

        while (remaining.isNotEmpty()) {
            var earliestIdx = -1
            var matchedSpecial: Pair<String, Int>? = null

            for (sp in specialTokens) {
                val idx = remaining.indexOf(sp.first)
                if (idx != -1 && (earliestIdx == -1 || idx < earliestIdx)) {
                    earliestIdx = idx
                    matchedSpecial = sp
                }
            }

            if (earliestIdx == -1) {
                result.addAll(encodeChunk(remaining))
                break
            } else {
                if (earliestIdx > 0) {
                    val prefix = remaining.substring(0, earliestIdx)
                    result.addAll(encodeChunk(prefix))
                }
                result.add(matchedSpecial!!.second)
                remaining = remaining.substring(earliestIdx + matchedSpecial.first.length)
            }
        }

        return result
    }

    /**
     * Formats conversation messages into standard ChatML and tokenizes the prompt.
     */
    fun applyChatTemplate(messages: List<Map<String, String>>, addGenerationPrompt: Boolean = true): List<Int> {
        val sb = StringBuilder()
        for (msg in messages) {
            val role = msg["role"] ?: "user"
            val content = msg["content"] ?: ""
            sb.append("<|im_start|>").append(role).append("\n")
            sb.append(content).append("<|im_end|>\n")
        }
        if (addGenerationPrompt) {
            sb.append("<|im_start|>assistant\n")
        }
        return encode(sb.toString())
    }

    /**
     * Decodes a single token ID into its UTF-8 text representation.
     */
    fun decodeToken(tokenId: Int): String {
        if (tokenId == IM_START_ID || tokenId == IM_END_ID || tokenId == ENDOFTEXT_ID) {
            return ""
        }
        val s = invVocab[tokenId] ?: return ""
        val bytes = ByteArray(s.length * 4)
        var byteCount = 0

        for (i in 0 until s.length) {
            val ch = s[i]
            val b = unicodeToByte[ch]
            if (b != null) {
                bytes[byteCount++] = b.toByte()
            } else {
                val cb = ch.toString().toByteArray(StandardCharsets.UTF_8)
                System.arraycopy(cb, 0, bytes, byteCount, cb.size)
                byteCount += cb.size
            }
        }
        return String(bytes, 0, byteCount, StandardCharsets.UTF_8)
    }

    /**
     * Decodes a list of token IDs into a human-readable UTF-8 string.
     */
    fun decode(tokenIds: List<Int>): String {
        val sb = StringBuilder()
        for (tid in tokenIds) {
            sb.append(decodeToken(tid))
        }
        return sb.toString()
    }
}
