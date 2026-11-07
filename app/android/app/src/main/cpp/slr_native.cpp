#include <jni.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <android/log.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define TAG "SLR_NATIVE_CPP"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static const uint8_t* g_safetensors_mmap = nullptr;
static size_t g_safetensors_size = 0;
static int g_safetensors_fd = -1;

#if defined(__ARM_NEON)
static inline float reduce_f32x4(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t p = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    p = vpadd_f32(p, p);
    return vget_lane_f32(p, 0);
#endif
}

static inline float32x4_t fma_f32x4(float32x4_t acc, float32x4_t a, float32x4_t b) {
#if defined(__aarch64__)
    return vfmaq_f32(acc, a, b);
#else
    return vmlaq_f32(acc, a, b);
#endif
}
#endif

extern "C" {

/**
 * High-performance RMSNorm kernel (ARM NEON accelerated with scalar fallback).
 */
JNIEXPORT void JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeRmsNorm(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jx,
    jfloatArray jweight,
    jfloatArray jout,
    jint len,
    jfloat eps
) {
    jfloat* x = env->GetFloatArrayElements(jx, nullptr);
    jfloat* weight = env->GetFloatArrayElements(jweight, nullptr);
    jfloat* out = env->GetFloatArrayElements(jout, nullptr);

    float totalSq = 0.0f;
    int i = 0;

#if defined(__ARM_NEON)
    float32x4_t sumSq0 = vdupq_n_f32(0.0f);
    float32x4_t sumSq1 = vdupq_n_f32(0.0f);
    int limit8 = len - 7;
    for (; i < limit8; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        sumSq0 = fma_f32x4(sumSq0, v0, v0);
        sumSq1 = fma_f32x4(sumSq1, v1, v1);
    }
    totalSq = reduce_f32x4(vaddq_f32(sumSq0, sumSq1));
#endif

    for (; i < len; ++i) {
        totalSq += x[i] * x[i];
    }

    float rms = std::sqrt(totalSq / len + eps);
    float invRms = 1.0f / rms;

    i = 0;
#if defined(__ARM_NEON)
    float32x4_t vInvRms = vdupq_n_f32(invRms);
    for (; i < limit8; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4_t w0 = vld1q_f32(weight + i);
        float32x4_t w1 = vld1q_f32(weight + i + 4);

        float32x4_t r0 = vmulq_f32(vmulq_f32(v0, vInvRms), w0);
        float32x4_t r1 = vmulq_f32(vmulq_f32(v1, vInvRms), w1);

        vst1q_f32(out + i, r0);
        vst1q_f32(out + i + 4, r1);
    }
#endif

    for (; i < len; ++i) {
        out[i] = (x[i] * invRms) * weight[i];
    }

    env->ReleaseFloatArrayElements(jx, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(jweight, weight, JNI_ABORT);
    env->ReleaseFloatArrayElements(jout, out, 0);
}

/**
 * Parallel 16-way unrolled GEMV: y = x * W^T
 * W shape: [outDim, inDim], row-major.
 */
JNIEXPORT void JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeGemv(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jx,
    jfloatArray jw,
    jfloatArray jy,
    jint inDim,
    jint outDim
) {
    jfloat* x = env->GetFloatArrayElements(jx, nullptr);
    jfloat* w = env->GetFloatArrayElements(jw, nullptr);
    jfloat* y = env->GetFloatArrayElements(jy, nullptr);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < outDim; ++i) {
        const float* row = w + (size_t)i * inDim;
        float total = 0.0f;
        int j = 0;

#if defined(__ARM_NEON)
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        int limit16 = inDim - 15;
        for (; j < limit16; j += 16) {
            float32x4_t xj0 = vld1q_f32(x + j);
            float32x4_t wj0 = vld1q_f32(row + j);
            sum0 = fma_f32x4(sum0, xj0, wj0);

            float32x4_t xj1 = vld1q_f32(x + j + 4);
            float32x4_t wj1 = vld1q_f32(row + j + 4);
            sum1 = fma_f32x4(sum1, xj1, wj1);

            float32x4_t xj2 = vld1q_f32(x + j + 8);
            float32x4_t wj2 = vld1q_f32(row + j + 8);
            sum2 = fma_f32x4(sum2, xj2, wj2);

            float32x4_t xj3 = vld1q_f32(x + j + 12);
            float32x4_t wj3 = vld1q_f32(row + j + 12);
            sum3 = fma_f32x4(sum3, xj3, wj3);
        }

        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);
        total = reduce_f32x4(sum0);
#endif

        for (; j < inDim; ++j) {
            total += x[j] * row[j];
        }
        y[i] = total;
    }

    env->ReleaseFloatArrayElements(jx, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(jw, w, JNI_ABORT);
    env->ReleaseFloatArrayElements(jy, y, 0);
}

/**
 * Parallel 16-way unrolled GEMV with Bias: y = x * W^T + b
 */
JNIEXPORT void JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeGemvAdd(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jx,
    jfloatArray jw,
    jfloatArray jb,
    jfloatArray jy,
    jint inDim,
    jint outDim
) {
    jfloat* x = env->GetFloatArrayElements(jx, nullptr);
    jfloat* w = env->GetFloatArrayElements(jw, nullptr);
    jfloat* b = env->GetFloatArrayElements(jb, nullptr);
    jfloat* y = env->GetFloatArrayElements(jy, nullptr);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < outDim; ++i) {
        const float* row = w + (size_t)i * inDim;
        float total = b[i];
        int j = 0;

#if defined(__ARM_NEON)
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        int limit16 = inDim - 15;
        for (; j < limit16; j += 16) {
            float32x4_t xj0 = vld1q_f32(x + j);
            float32x4_t wj0 = vld1q_f32(row + j);
            sum0 = fma_f32x4(sum0, xj0, wj0);

            float32x4_t xj1 = vld1q_f32(x + j + 4);
            float32x4_t wj1 = vld1q_f32(row + j + 4);
            sum1 = fma_f32x4(sum1, xj1, wj1);

            float32x4_t xj2 = vld1q_f32(x + j + 8);
            float32x4_t wj2 = vld1q_f32(row + j + 8);
            sum2 = fma_f32x4(sum2, xj2, wj2);

            float32x4_t xj3 = vld1q_f32(x + j + 12);
            float32x4_t wj3 = vld1q_f32(row + j + 12);
            sum3 = fma_f32x4(sum3, xj3, wj3);
        }

        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);
        total += reduce_f32x4(sum0);
#endif

        for (; j < inDim; ++j) {
            total += x[j] * row[j];
        }
        y[i] = total;
    }

    env->ReleaseFloatArrayElements(jx, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(jw, w, JNI_ABORT);
    env->ReleaseFloatArrayElements(jb, b, JNI_ABORT);
    env->ReleaseFloatArrayElements(jy, y, 0);
}

/**
 * Parallel Direct BF16 GEMV: y = x * W^T
 * Reads row weights directly from memory-mapped safetensors file without unpacking into RAM.
 * Unrolls 16-way using ARM NEON registers.
 */
JNIEXPORT void JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeGemvBf16(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jx,
    jlong weightOffsetBytes,
    jfloatArray jy,
    jint inDim,
    jint outDim
) {
    if (!g_safetensors_mmap || weightOffsetBytes < 0) return;
    jfloat* x = env->GetFloatArrayElements(jx, nullptr);
    jfloat* y = env->GetFloatArrayElements(jy, nullptr);
    const uint16_t* w = (const uint16_t*)(g_safetensors_mmap + weightOffsetBytes);

    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < outDim; ++i) {
        const uint16_t* row = w + (size_t)i * inDim;
        float total = 0.0f;
        int j = 0;

#if defined(__ARM_NEON)
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        int limit16 = inDim - 15;
        for (; j < limit16; j += 16) {
            uint16x8_t bf16_a = vld1q_u16(row + j);
            uint16x8_t bf16_b = vld1q_u16(row + j + 8);

            uint32x4_t fp32_u0 = vshll_n_u16(vget_low_u16(bf16_a), 16);
            uint32x4_t fp32_u1 = vshll_n_u16(vget_high_u16(bf16_a), 16);
            uint32x4_t fp32_u2 = vshll_n_u16(vget_low_u16(bf16_b), 16);
            uint32x4_t fp32_u3 = vshll_n_u16(vget_high_u16(bf16_b), 16);

            float32x4_t w0 = vreinterpretq_f32_u32(fp32_u0);
            float32x4_t w1 = vreinterpretq_f32_u32(fp32_u1);
            float32x4_t w2 = vreinterpretq_f32_u32(fp32_u2);
            float32x4_t w3 = vreinterpretq_f32_u32(fp32_u3);

            float32x4_t xj0 = vld1q_f32(x + j);
            float32x4_t xj1 = vld1q_f32(x + j + 4);
            float32x4_t xj2 = vld1q_f32(x + j + 8);
            float32x4_t xj3 = vld1q_f32(x + j + 12);

            sum0 = fma_f32x4(sum0, xj0, w0);
            sum1 = fma_f32x4(sum1, xj1, w1);
            sum2 = fma_f32x4(sum2, xj2, w2);
            sum3 = fma_f32x4(sum3, xj3, w3);
        }

        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);
        total += reduce_f32x4(sum0);
#endif

        for (; j < inDim; ++j) {
            uint32_t bits = ((uint32_t)row[j]) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(float));
            total += x[j] * f;
        }
        y[i] = total;
    }

    env->ReleaseFloatArrayElements(jx, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(jy, y, 0);
}

/**
 * Parallel Direct BF16 GEMV with Bias: y = x * W^T + b
 * Reads row weights directly from memory-mapped safetensors file without unpacking into RAM.
 * Unrolls 16-way using ARM NEON registers.
 */
JNIEXPORT void JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeGemvAddBf16(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jx,
    jlong weightOffsetBytes,
    jfloatArray jb,
    jfloatArray jy,
    jint inDim,
    jint outDim
) {
    if (!g_safetensors_mmap || weightOffsetBytes < 0) return;
    jfloat* x = env->GetFloatArrayElements(jx, nullptr);
    jfloat* b = env->GetFloatArrayElements(jb, nullptr);
    jfloat* y = env->GetFloatArrayElements(jy, nullptr);
    const uint16_t* w = (const uint16_t*)(g_safetensors_mmap + weightOffsetBytes);

    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < outDim; ++i) {
        const uint16_t* row = w + (size_t)i * inDim;
        float total = b[i];
        int j = 0;

#if defined(__ARM_NEON)
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        int limit16 = inDim - 15;
        for (; j < limit16; j += 16) {
            uint16x8_t bf16_a = vld1q_u16(row + j);
            uint16x8_t bf16_b = vld1q_u16(row + j + 8);

            uint32x4_t fp32_u0 = vshll_n_u16(vget_low_u16(bf16_a), 16);
            uint32x4_t fp32_u1 = vshll_n_u16(vget_high_u16(bf16_a), 16);
            uint32x4_t fp32_u2 = vshll_n_u16(vget_low_u16(bf16_b), 16);
            uint32x4_t fp32_u3 = vshll_n_u16(vget_high_u16(bf16_b), 16);

            float32x4_t w0 = vreinterpretq_f32_u32(fp32_u0);
            float32x4_t w1 = vreinterpretq_f32_u32(fp32_u1);
            float32x4_t w2 = vreinterpretq_f32_u32(fp32_u2);
            float32x4_t w3 = vreinterpretq_f32_u32(fp32_u3);

            float32x4_t xj0 = vld1q_f32(x + j);
            float32x4_t xj1 = vld1q_f32(x + j + 4);
            float32x4_t xj2 = vld1q_f32(x + j + 8);
            float32x4_t xj3 = vld1q_f32(x + j + 12);

            sum0 = fma_f32x4(sum0, xj0, w0);
            sum1 = fma_f32x4(sum1, xj1, w1);
            sum2 = fma_f32x4(sum2, xj2, w2);
            sum3 = fma_f32x4(sum3, xj3, w3);
        }

        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);
        total += reduce_f32x4(sum0);
#endif

        for (; j < inDim; ++j) {
            uint32_t bits = ((uint32_t)row[j]) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(float));
            total += x[j] * f;
        }
        y[i] = total;
    }

    env->ReleaseFloatArrayElements(jx, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(jb, b, JNI_ABORT);
    env->ReleaseFloatArrayElements(jy, y, 0);
}

/**
 * Parallel Direct Batched BF16 GEMV: Y = X * W^T
 * Evaluates row weights directly from memory-mapped safetensors across all batchSize tokens
 * keeping row in L1 cache during multi-token multiplication.
 */
JNIEXPORT void JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeBatchedGemvBf16(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jxBatch,
    jlong weightOffsetBytes,
    jfloatArray jyBatch,
    jint inDim,
    jint outDim,
    jint batchSize
) {
    if (!g_safetensors_mmap || weightOffsetBytes < 0 || batchSize <= 0) return;
    jfloat* xBatch = env->GetFloatArrayElements(jxBatch, nullptr);
    jfloat* yBatch = env->GetFloatArrayElements(jyBatch, nullptr);
    const uint16_t* w = (const uint16_t*)(g_safetensors_mmap + weightOffsetBytes);

    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < outDim; ++i) {
        const uint16_t* row = w + (size_t)i * inDim;

        for (int t = 0; t < batchSize; ++t) {
            const float* xt = xBatch + (size_t)t * inDim;
            float total = 0.0f;
            int j = 0;

#if defined(__ARM_NEON)
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);

            int limit16 = inDim - 15;
            for (; j < limit16; j += 16) {
                uint16x8_t bf16_a = vld1q_u16(row + j);
                uint16x8_t bf16_b = vld1q_u16(row + j + 8);

                uint32x4_t fp32_u0 = vshll_n_u16(vget_low_u16(bf16_a), 16);
                uint32x4_t fp32_u1 = vshll_n_u16(vget_high_u16(bf16_a), 16);
                uint32x4_t fp32_u2 = vshll_n_u16(vget_low_u16(bf16_b), 16);
                uint32x4_t fp32_u3 = vshll_n_u16(vget_high_u16(bf16_b), 16);

                float32x4_t w0 = vreinterpretq_f32_u32(fp32_u0);
                float32x4_t w1 = vreinterpretq_f32_u32(fp32_u1);
                float32x4_t w2 = vreinterpretq_f32_u32(fp32_u2);
                float32x4_t w3 = vreinterpretq_f32_u32(fp32_u3);

                float32x4_t xj0 = vld1q_f32(xt + j);
                float32x4_t xj1 = vld1q_f32(xt + j + 4);
                float32x4_t xj2 = vld1q_f32(xt + j + 8);
                float32x4_t xj3 = vld1q_f32(xt + j + 12);

                sum0 = fma_f32x4(sum0, xj0, w0);
                sum1 = fma_f32x4(sum1, xj1, w1);
                sum2 = fma_f32x4(sum2, xj2, w2);
                sum3 = fma_f32x4(sum3, xj3, w3);
            }

            sum0 = vaddq_f32(sum0, sum1);
            sum2 = vaddq_f32(sum2, sum3);
            sum0 = vaddq_f32(sum0, sum2);
            total += reduce_f32x4(sum0);
#endif

            for (; j < inDim; ++j) {
                uint32_t bits = ((uint32_t)row[j]) << 16;
                float f;
                std::memcpy(&f, &bits, sizeof(float));
                total += xt[j] * f;
            }
            yBatch[(size_t)t * outDim + i] = total;
        }
    }

    env->ReleaseFloatArrayElements(jxBatch, xBatch, JNI_ABORT);
    env->ReleaseFloatArrayElements(jyBatch, yBatch, 0);
}

/**
 * Parallel Direct Batched BF16 GEMV with Bias: Y = X * W^T + b
 * Evaluates row weights directly from memory-mapped safetensors across all batchSize tokens.
 */
JNIEXPORT void JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeBatchedGemvAddBf16(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jxBatch,
    jlong weightOffsetBytes,
    jfloatArray jb,
    jfloatArray jyBatch,
    jint inDim,
    jint outDim,
    jint batchSize
) {
    if (!g_safetensors_mmap || weightOffsetBytes < 0 || batchSize <= 0) return;
    jfloat* xBatch = env->GetFloatArrayElements(jxBatch, nullptr);
    jfloat* b = env->GetFloatArrayElements(jb, nullptr);
    jfloat* yBatch = env->GetFloatArrayElements(jyBatch, nullptr);
    const uint16_t* w = (const uint16_t*)(g_safetensors_mmap + weightOffsetBytes);

    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < outDim; ++i) {
        const uint16_t* row = w + (size_t)i * inDim;
        float biasVal = b[i];

        for (int t = 0; t < batchSize; ++t) {
            const float* xt = xBatch + (size_t)t * inDim;
            float total = biasVal;
            int j = 0;

#if defined(__ARM_NEON)
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);

            int limit16 = inDim - 15;
            for (; j < limit16; j += 16) {
                uint16x8_t bf16_a = vld1q_u16(row + j);
                uint16x8_t bf16_b = vld1q_u16(row + j + 8);

                uint32x4_t fp32_u0 = vshll_n_u16(vget_low_u16(bf16_a), 16);
                uint32x4_t fp32_u1 = vshll_n_u16(vget_high_u16(bf16_a), 16);
                uint32x4_t fp32_u2 = vshll_n_u16(vget_low_u16(bf16_b), 16);
                uint32x4_t fp32_u3 = vshll_n_u16(vget_high_u16(bf16_b), 16);

                float32x4_t w0 = vreinterpretq_f32_u32(fp32_u0);
                float32x4_t w1 = vreinterpretq_f32_u32(fp32_u1);
                float32x4_t w2 = vreinterpretq_f32_u32(fp32_u2);
                float32x4_t w3 = vreinterpretq_f32_u32(fp32_u3);

                float32x4_t xj0 = vld1q_f32(xt + j);
                float32x4_t xj1 = vld1q_f32(xt + j + 4);
                float32x4_t xj2 = vld1q_f32(xt + j + 8);
                float32x4_t xj3 = vld1q_f32(xt + j + 12);

                sum0 = fma_f32x4(sum0, xj0, w0);
                sum1 = fma_f32x4(sum1, xj1, w1);
                sum2 = fma_f32x4(sum2, xj2, w2);
                sum3 = fma_f32x4(sum3, xj3, w3);
            }

            sum0 = vaddq_f32(sum0, sum1);
            sum2 = vaddq_f32(sum2, sum3);
            sum0 = vaddq_f32(sum0, sum2);
            total += reduce_f32x4(sum0);
#endif

            for (; j < inDim; ++j) {
                uint32_t bits = ((uint32_t)row[j]) << 16;
                float f;
                std::memcpy(&f, &bits, sizeof(float));
                total += xt[j] * f;
            }
            yBatch[(size_t)t * outDim + i] = total;
        }
    }

    env->ReleaseFloatArrayElements(jxBatch, xBatch, JNI_ABORT);
    env->ReleaseFloatArrayElements(jb, b, JNI_ABORT);
    env->ReleaseFloatArrayElements(jyBatch, yBatch, 0);
}

/**
 * Vectorized SwiGLU Elementwise operation: mlpHidden = silu(gate) * up
 */
JNIEXPORT void JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeSwigluElementwise(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jgate,
    jfloatArray jup,
    jfloatArray jmlpHidden,
    jint intermediateSize
) {
    jfloat* gate = env->GetFloatArrayElements(jgate, nullptr);
    jfloat* up = env->GetFloatArrayElements(jup, nullptr);
    jfloat* mlpHidden = env->GetFloatArrayElements(jmlpHidden, nullptr);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < intermediateSize; ++i) {
        float g = gate[i];
        float silu_g = g / (1.0f + std::exp(-g));
        mlpHidden[i] = silu_g * up[i];
    }

    env->ReleaseFloatArrayElements(jgate, gate, JNI_ABORT);
    env->ReleaseFloatArrayElements(jup, up, JNI_ABORT);
    env->ReleaseFloatArrayElements(jmlpHidden, mlpHidden, 0);
}

/**
 * High-Speed Parallel BF16-to-FP32 Unpacker directly into reusable layer buffer.
 */
JNIEXPORT void JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeUnpackBf16ToFp32(
    JNIEnv* env,
    jobject /* this */,
    jobject srcDirectBuf,
    jlong srcOffsetBytes,
    jfloatArray jdst,
    jint numElements
) {
    const uint8_t* base = (const uint8_t*)env->GetDirectBufferAddress(srcDirectBuf);
    if (!base) return;
    const uint16_t* src = (const uint16_t*)(base + srcOffsetBytes);
    jfloat* dst = env->GetFloatArrayElements(jdst, nullptr);

    int idx = 0;

#if defined(__ARM_NEON)
    int numChunks = numElements / 16;
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < numChunks; ++c) {
        int i = c * 16;
        uint16x8_t bf16_a = vld1q_u16(src + i);
        uint16x8_t bf16_b = vld1q_u16(src + i + 8);

        uint32x4_t fp32_u0 = vshll_n_u16(vget_low_u16(bf16_a), 16);
        uint32x4_t fp32_u1 = vshll_n_u16(vget_high_u16(bf16_a), 16);
        uint32x4_t fp32_u2 = vshll_n_u16(vget_low_u16(bf16_b), 16);
        uint32x4_t fp32_u3 = vshll_n_u16(vget_high_u16(bf16_b), 16);

        vst1q_f32(dst + i, vreinterpretq_f32_u32(fp32_u0));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(fp32_u1));
        vst1q_f32(dst + i + 8, vreinterpretq_f32_u32(fp32_u2));
        vst1q_f32(dst + i + 12, vreinterpretq_f32_u32(fp32_u3));
    }
    idx = numChunks * 16;
#endif

    for (int i = idx; i < numElements; ++i) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        dst[i] = f;
    }

    env->ReleaseFloatArrayElements(jdst, dst, 0);
}

/**
 * High-Speed Parallel Argmax Output Projection across entire vocabulary (151,936 tokens).
 * Computes dot-product of normed (1536) directly against BF16 memory-mapped embeddings.
 * Zero-allocation, fully vectorized with ARM NEON.
 */
JNIEXPORT jint JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeArgmaxVocabProjection(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jnormed,
    jobject embedDirectBuf,
    jlong embedOffsetBytes,
    jint vocabSize,
    jint hiddenSize
) {
    jfloat* normed = env->GetFloatArrayElements(jnormed, nullptr);
    const uint8_t* base = (const uint8_t*)env->GetDirectBufferAddress(embedDirectBuf);
    if (!base) {
        env->ReleaseFloatArrayElements(jnormed, normed, JNI_ABORT);
        return 0;
    }
    const uint16_t* embed = (const uint16_t*)(base + embedOffsetBytes);

    int maxThreads = omp_get_max_threads();
    std::vector<float> threadMaxLogit(maxThreads, -1e30f);
    std::vector<int> threadBestToken(maxThreads, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float localMax = -1e30f;
        int localBest = 0;

        #pragma omp for schedule(static)
        for (int t = 0; t < vocabSize; ++t) {
            const uint16_t* row = embed + (size_t)t * hiddenSize;
            float total = 0.0f;
            int d = 0;

#if defined(__ARM_NEON)
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            int limit8 = hiddenSize - 7;
            for (; d < limit8; d += 8) {
                uint16x8_t bf16_8 = vld1q_u16(row + d);
                uint32x4_t fp32_u0 = vshll_n_u16(vget_low_u16(bf16_8), 16);
                uint32x4_t fp32_u1 = vshll_n_u16(vget_high_u16(bf16_8), 16);
                float32x4_t w0 = vreinterpretq_f32_u32(fp32_u0);
                float32x4_t w1 = vreinterpretq_f32_u32(fp32_u1);

                float32x4_t n0 = vld1q_f32(normed + d);
                float32x4_t n1 = vld1q_f32(normed + d + 4);

                sum0 = fma_f32x4(sum0, n0, w0);
                sum1 = fma_f32x4(sum1, n1, w1);
            }
            total = reduce_f32x4(vaddq_f32(sum0, sum1));
#endif

            for (; d < hiddenSize; ++d) {
                uint32_t bits = ((uint32_t)row[d]) << 16;
                float f;
                std::memcpy(&f, &bits, sizeof(float));
                total += normed[d] * f;
            }

            if (total > localMax) {
                localMax = total;
                localBest = t;
            }
        }

        threadMaxLogit[tid] = localMax;
        threadBestToken[tid] = localBest;
    }

    float globalMax = threadMaxLogit[0];
    int globalBest = threadBestToken[0];
    for (int i = 1; i < maxThreads; ++i) {
        if (threadMaxLogit[i] > globalMax) {
            globalMax = threadMaxLogit[i];
            globalBest = threadBestToken[i];
        }
    }

    env->ReleaseFloatArrayElements(jnormed, normed, JNI_ABORT);
    return globalBest;
}

JNIEXPORT jboolean JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeInitSafetensors(
    JNIEnv* env,
    jobject /* this */,
    jstring jpath
) {
    if (g_safetensors_mmap != nullptr) {
        return JNI_TRUE;
    }
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) return JNI_FALSE;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        LOGE("Failed to open safetensors file %s: errno=%d", path, errno);
        env->ReleaseStringUTFChars(jpath, path);
        return JNI_FALSE;
    }
    env->ReleaseStringUTFChars(jpath, path);

    struct stat st;
    if (fstat(fd, &st) != 0) {
        LOGE("Failed to fstat safetensors file");
        close(fd);
        return JNI_FALSE;
    }

    size_t fileSize = (size_t)st.st_size;
    void* ptr = mmap(NULL, fileSize, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        LOGE("Failed to mmap safetensors file of size %zu: errno=%d", fileSize, errno);
        close(fd);
        return JNI_FALSE;
    }

    g_safetensors_mmap = (const uint8_t*)ptr;
    g_safetensors_size = fileSize;
    g_safetensors_fd = fd;
    LOGI("Successfully mapped safetensors file (%zu bytes, %.2f MB)", fileSize, fileSize / (1024.0 * 1024.0));
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeCloseSafetensors(
    JNIEnv* /* env */,
    jobject /* this */
) {
    if (g_safetensors_mmap != nullptr) {
        munmap((void*)g_safetensors_mmap, g_safetensors_size);
        g_safetensors_mmap = nullptr;
        g_safetensors_size = 0;
    }
    if (g_safetensors_fd >= 0) {
        close(g_safetensors_fd);
        g_safetensors_fd = -1;
    }
    LOGI("Safetensors mmap closed.");
}

JNIEXPORT void JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeEvictRange(
    JNIEnv* /* env */,
    jobject /* this */,
    jlong offsetBytes,
    jlong lengthBytes
) {
    if (!g_safetensors_mmap || offsetBytes < 0 || lengthBytes <= 0) return;
    if ((size_t)(offsetBytes + lengthBytes) > g_safetensors_size) return;

    long pageSize = sysconf(_SC_PAGESIZE);
    long pageMask = ~(pageSize - 1);
    uintptr_t start = (uintptr_t)(g_safetensors_mmap + offsetBytes);
    uintptr_t alignedStart = start & pageMask;
    size_t alignedLength = (start + lengthBytes) - alignedStart;

    madvise((void*)alignedStart, alignedLength, MADV_DONTNEED);
}

JNIEXPORT void JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeUnpackBf16FromOffset(
    JNIEnv* env,
    jobject /* this */,
    jlong offsetBytes,
    jfloatArray jdst,
    jint numElements
) {
    if (!g_safetensors_mmap || offsetBytes < 0) return;
    const uint16_t* src = (const uint16_t*)(g_safetensors_mmap + offsetBytes);
    jfloat* dst = env->GetFloatArrayElements(jdst, nullptr);

    int idx = 0;
#if defined(__ARM_NEON)
    int numChunks = numElements / 16;
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < numChunks; ++c) {
        int i = c * 16;
        uint16x8_t bf16_a = vld1q_u16(src + i);
        uint16x8_t bf16_b = vld1q_u16(src + i + 8);

        uint32x4_t fp32_u0 = vshll_n_u16(vget_low_u16(bf16_a), 16);
        uint32x4_t fp32_u1 = vshll_n_u16(vget_high_u16(bf16_a), 16);
        uint32x4_t fp32_u2 = vshll_n_u16(vget_low_u16(bf16_b), 16);
        uint32x4_t fp32_u3 = vshll_n_u16(vget_high_u16(bf16_b), 16);

        vst1q_f32(dst + i, vreinterpretq_f32_u32(fp32_u0));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(fp32_u1));
        vst1q_f32(dst + i + 8, vreinterpretq_f32_u32(fp32_u2));
        vst1q_f32(dst + i + 12, vreinterpretq_f32_u32(fp32_u3));
    }
    idx = numChunks * 16;
#endif

    for (int i = idx; i < numElements; ++i) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        dst[i] = f;
    }

    env->ReleaseFloatArrayElements(jdst, dst, 0);
}

JNIEXPORT jint JNICALL
Java_com_example_exp07_1slr_AndroidSlrEngine_nativeArgmaxVocabProjectionDirect(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray jnormed,
    jlong embedOffsetBytes,
    jint vocabSize,
    jint hiddenSize
) {
    if (!g_safetensors_mmap || embedOffsetBytes < 0) return 0;
    jfloat* normed = env->GetFloatArrayElements(jnormed, nullptr);
    const uint16_t* embed = (const uint16_t*)(g_safetensors_mmap + embedOffsetBytes);

    int maxThreads = omp_get_max_threads();
    std::vector<float> threadMaxLogit(maxThreads, -1e30f);
    std::vector<int> threadBestToken(maxThreads, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float localMax = -1e30f;
        int localBest = 0;

        #pragma omp for schedule(static)
        for (int t = 0; t < vocabSize; ++t) {
            const uint16_t* row = embed + (size_t)t * hiddenSize;
            float total = 0.0f;
            int d = 0;

#if defined(__ARM_NEON)
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            int limit8 = hiddenSize - 7;
            for (; d < limit8; d += 8) {
                uint16x8_t bf16_8 = vld1q_u16(row + d);
                uint32x4_t fp32_u0 = vshll_n_u16(vget_low_u16(bf16_8), 16);
                uint32x4_t fp32_u1 = vshll_n_u16(vget_high_u16(bf16_8), 16);
                float32x4_t w0 = vreinterpretq_f32_u32(fp32_u0);
                float32x4_t w1 = vreinterpretq_f32_u32(fp32_u1);

                float32x4_t n0 = vld1q_f32(normed + d);
                float32x4_t n1 = vld1q_f32(normed + d + 4);

                sum0 = fma_f32x4(sum0, n0, w0);
                sum1 = fma_f32x4(sum1, n1, w1);
            }
            total = reduce_f32x4(vaddq_f32(sum0, sum1));
#endif

            for (; d < hiddenSize; ++d) {
                uint32_t bits = ((uint32_t)row[d]) << 16;
                float f;
                std::memcpy(&f, &bits, sizeof(float));
                total += normed[d] * f;
            }

            if (total > localMax) {
                localMax = total;
                localBest = t;
            }
        }

        threadMaxLogit[tid] = localMax;
        threadBestToken[tid] = localBest;
    }

    float globalMax = threadMaxLogit[0];
    int globalBest = threadBestToken[0];
    for (int i = 1; i < maxThreads; ++i) {
        if (threadMaxLogit[i] > globalMax) {
            globalMax = threadMaxLogit[i];
            globalBest = threadBestToken[i];
        }
    }

    env->ReleaseFloatArrayElements(jnormed, normed, JNI_ABORT);
    return globalBest;
}

} // extern "C"
