// SPDX-License-Identifier: GPL-2.0-only
/*
 * Simple BF16 dense GEMM with FP32 accumulation, to confirm a gpu-unlock.
 *
 * The unlock doubles FP16/BF16/TF32 tensor-core throughput with FP32
 * accumulation.  A locked card runs these at half rate, an unlocked card at
 * full rate.  Compare the measured TFLOPS against the README "before"/"after"
 * table for your card.
 *
 * Build (requires CUDA toolkit + cuBLAS):
 *   nvcc -O3 -arch=sm_86 gemm_fp16.cu -lcublas -o gemm_bf16
 *
 * Run:
 *   ./gemm_bf16 [size] [iters]
 *
 * Example (RTX 3080, unlocked):  ./gemm_bf16 8192 20
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), \
                __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 8192;
    int iters = argc > 2 ? atoi(argv[2]) : 20;
    if (n <= 0 || iters <= 0) {
        fprintf(stderr, "usage: %s [size] [iters]\n", argv[0]);
        return 1;
    }

    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);

    cublasHandle_t h;
    CHECK_CUBLAS(cublasCreate(&h));

    size_t nbytes = (size_t)n * n;
    size_t abytes = nbytes * sizeof(__nv_bfloat16);
    size_t cbytes = nbytes * sizeof(float);
    __nv_bfloat16 *A, *B;
    float *C;
    CHECK_CUDA(cudaMalloc(&A, abytes));
    CHECK_CUDA(cudaMalloc(&B, abytes));
    CHECK_CUDA(cudaMalloc(&C, cbytes));
    CHECK_CUDA(cudaMemset(C, 0, cbytes));

    // Fill with small values to avoid overflow in accumulation.
    __nv_bfloat16 *hA = (__nv_bfloat16 *)malloc(abytes);
    __nv_bfloat16 *hB = (__nv_bfloat16 *)malloc(abytes);
    for (size_t i = 0; i < nbytes; i++) {
        hA[i] = __float2bfloat16(0.5f);
        hB[i] = __float2bfloat16(0.5f);
    }
    CHECK_CUDA(cudaMemcpy(A, hA, abytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B, hB, abytes, cudaMemcpyHostToDevice));

    // BF16 inputs, FP32 accumulation (the rate the unlock doubles).
    __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    float beta = 0.0f;

    // Warm up.
    CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                              A, CUDA_R_16BF, n, B, CUDA_R_16BF, n, &beta,
                              C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Time the GEMMs.
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                  A, CUDA_R_16BF, n, B, CUDA_R_16BF, n, &beta,
                                  C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    double flops = 2.0 * (double)n * n * n * iters;
    double tflops = flops / (ms * 1e-3) / 1e12;

    printf("GEMM %dx%d BF16 (FP32 accum), %d iters: %.1f ms total (%.2f ms each)\n",
           n, n, iters, ms, ms / iters);
    printf("BF16 tensor throughput: %.1f TFLOPS\n", tflops);
    printf("\nCompare against the README table for your card.\n");
    printf("If this is ~2x the locked value, the card is unlocked.\n");

    free(hA);
    free(hB);
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(C));
    CHECK_CUBLAS(cublasDestroy(h));
    return 0;
}