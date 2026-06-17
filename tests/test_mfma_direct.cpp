/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE in the project root for license information.
 */

// Direct MFMA test - uses inline assembly to execute MFMA instructions
// This validates the fp64_monitor tool's GEMM analysis feature
//
// Note: This test uses direct MFMA assembly rather than rocBLAS because
// rocBLAS can conflict with rocprofiler-sdk when loaded via ROCP_TOOL_LIBRARIES.
//
// Build: hipcc -o test_mfma_direct test_mfma_direct.cpp
// Run:   ./test_mfma_direct [iterations_per_thread] [kernel_launches]
//
// Expected validation:
//   - FP64 MFMA counter should match "Total MFMA (counter)" output
//   - TFLOPS should be approximately equal to "Achieved" output
//   - GEMM Likelihood should be "Very High"

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

#define HIP_CHECK(cmd) do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(error) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// Kernel that executes FP64 MFMA instructions
// V_MFMA_F64_4X4X4F64: 4x4x4 matrix multiply = 64 FMA = 128 FLOPs per instruction
__global__ void mfma_fp64_kernel(double* C, const double* A, const double* B, 
                                  int iterations_per_thread) {
    // Each thread executes MFMA instructions in a loop
    // We use inline assembly for direct MFMA control
    
    double a = A[threadIdx.x % 64];
    double b = B[threadIdx.x % 64];
    double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
    
    // Execute many MFMA instructions
    for (int i = 0; i < iterations_per_thread; i++) {
        // V_MFMA_F64_4X4X4F64 computes a 4x4x4 matrix multiply
        // Each instruction = 64 FMA operations = 128 FLOPs
        // Using inline assembly for gfx90a/gfx942
        #if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        asm volatile(
            "v_mfma_f64_4x4x4f64 %0, %4, %5, %0\n"
            "v_mfma_f64_4x4x4f64 %1, %4, %5, %1\n"
            "v_mfma_f64_4x4x4f64 %2, %4, %5, %2\n"
            "v_mfma_f64_4x4x4f64 %3, %4, %5, %3\n"
            : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3)
            : "v"(a), "v"(b)
        );
        #else
        // Fallback for non-MFMA GPUs - use regular FMA
        c0 = fma(a, b, c0);
        c1 = fma(a, b, c1);
        c2 = fma(a, b, c2);
        c3 = fma(a, b, c3);
        #endif
    }
    
    // Prevent optimization
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        C[0] = c0 + c1 + c2 + c3;
    }
}

// Simpler kernel using regular FP64 FMA for comparison
__global__ void fma_fp64_kernel(double* C, const double* A, const double* B,
                                 int iterations_per_thread) {
    double a = A[threadIdx.x % 64];
    double b = B[threadIdx.x % 64];
    double c = 0.0;
    
    for (int i = 0; i < iterations_per_thread; i++) {
        c = fma(a, b, c);
        c = fma(a, b, c);
        c = fma(a, b, c);
        c = fma(a, b, c);
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        C[0] = c;
    }
}

int main(int argc, char* argv[]) {
    int iterations = 1000;      // Loop iterations per thread
    int num_blocks = 1024;      // Number of thread blocks
    int threads_per_block = 256;
    int kernel_launches = 100;  // Number of kernel launches
    
    if (argc > 1) iterations = std::atoi(argv[1]);
    if (argc > 2) kernel_launches = std::atoi(argv[2]);
    
    std::cout << "########################################################\n";
    std::cout << "#           Direct MFMA FP64 Test                      #\n";
    std::cout << "########################################################\n\n";
    
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "Device: " << props.name << "\n";
    std::cout << "  Compute Units: " << props.multiProcessorCount << "\n\n";
    
    std::cout << "Configuration:\n";
    std::cout << "  Blocks:        " << num_blocks << "\n";
    std::cout << "  Threads/block: " << threads_per_block << "\n";
    std::cout << "  Iterations:    " << iterations << " per thread\n";
    std::cout << "  Kernel runs:   " << kernel_launches << "\n\n";
    
    // Calculate expected MFMA instructions and FLOPs
    // Hardware counters count per-WAVEFRONT, not per-thread
    // A wavefront is 64 threads that execute MFMA together
    int wavefront_size = 64;
    long long wavefronts_per_block = threads_per_block / wavefront_size;
    long long total_wavefronts = (long long)num_blocks * wavefronts_per_block * kernel_launches;
    long long mfma_per_wavefront = 4LL * iterations;  // 4 MFMA instructions per iteration
    long long total_mfma_counter = total_wavefronts * mfma_per_wavefront;  // What counter will show
    long long flops_per_mfma = 128;  // 4x4x4 = 64 FMA = 128 FLOPs per MFMA
    long long total_flops = total_mfma_counter * flops_per_mfma;
    
    std::cout << "Expected Hardware Counter Values:\n";
    std::cout << "  Wavefronts/block: " << wavefronts_per_block << "\n";
    std::cout << "  MFMA/wavefront:   " << mfma_per_wavefront << "\n";
    std::cout << "  Total MFMA (counter): " << std::scientific << std::setprecision(3) 
              << (double)total_mfma_counter << "\n";
    std::cout << "  FLOPs/MFMA:       " << flops_per_mfma << "\n";
    std::cout << "  Total FLOPs:      " << (double)total_flops << "\n";
    std::cout << "  Total TFLOPs:     " << std::fixed << std::setprecision(3) 
              << total_flops / 1e12 << "\n\n";
    
    // Allocate memory
    double *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, 64 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B, 64 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C, sizeof(double)));
    
    // Initialize with 1.0
    double ones[64];
    for (int i = 0; i < 64; i++) ones[i] = 1.0;
    HIP_CHECK(hipMemcpy(d_A, ones, 64 * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, ones, 64 * sizeof(double), hipMemcpyHostToDevice));
    
    // Warmup
    std::cout << "Running warmup...\n";
    mfma_fp64_kernel<<<num_blocks, threads_per_block>>>(d_C, d_A, d_B, iterations);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Timed run
    std::cout << "Running " << kernel_launches << " kernel launches...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < kernel_launches; i++) {
        mfma_fp64_kernel<<<num_blocks, threads_per_block>>>(d_C, d_A, d_B, iterations);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    double achieved_tflops = total_flops / elapsed.count() / 1e12;
    
    std::cout << "======================================================================\n";
    std::cout << "Results\n";
    std::cout << "======================================================================\n";
    std::cout << "  Time:          " << std::fixed << std::setprecision(3) 
              << elapsed.count() * 1000 << " ms\n";
    std::cout << "  Achieved:      " << std::setprecision(2) 
              << achieved_tflops << " TFLOPS\n\n";
    
    std::cout << "Validation for fp64_monitor:\n";
    std::cout << "  Expected FP64 MFMA counter: ~" << std::scientific 
              << (double)total_mfma_counter << "\n";
    std::cout << "  Expected TFLOPS:            ~" << std::fixed << std::setprecision(2)
              << achieved_tflops << "\n";
    std::cout << "  GEMM Likelihood:            Very High (should be >50% MFMA)\n\n";
    
    std::cout << "########################################################\n";
    std::cout << "#                    Test Complete                      #\n";
    std::cout << "########################################################\n";
    
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    
    return 0;
}
