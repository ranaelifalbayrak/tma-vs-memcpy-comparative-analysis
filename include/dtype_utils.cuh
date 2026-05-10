#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ======================================================
// Type names
// ======================================================

template<typename T>
struct TypeName;

template<>
struct TypeName<float> {
    static constexpr const char* value = "fp32";
};

template<>
struct TypeName<__half> {
    static constexpr const char* value = "fp16";
};

template<>
struct TypeName<__nv_bfloat16> {
    static constexpr const char* value = "bf16";
};

template<>
struct TypeName<__nv_fp8_e4m3> {
    static constexpr const char* value = "fp8_e4m3";
};

// ======================================================
// To float
// ======================================================

template<typename T>
__device__ inline float to_float(T x);

template<>
__device__ inline float to_float<float>(float x) {
    return x;
}

template<>
__device__ inline float to_float<__half>(__half x) {
    return __half2float(x);
}

template<>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template<>
__device__ inline float to_float<__nv_fp8_e4m3>(__nv_fp8_e4m3 x) {
#if (__CUDA_ARCH__ >= 900)
    return __half2float((__half)x);
#else
    return 0.f;
#endif
}

// ======================================================
// From float
// ======================================================

template<typename T>
__device__ inline T from_float(float x);

template<>
__device__ inline float from_float<float>(float x) {
    return x;
}

template<>
__device__ inline __half from_float<__half>(float x) {
    return __float2half(x);
}

template<>
__device__ inline __nv_bfloat16
from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template<>
__device__ inline __nv_fp8_e4m3
from_float<__nv_fp8_e4m3>(float x) {
#if (__CUDA_ARCH__ >= 900)
    return (__nv_fp8_e4m3)x;
#else
    return (__nv_fp8_e4m3)0;
#endif
}