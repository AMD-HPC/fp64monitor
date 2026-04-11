// Test rocBLAS DGEMM for FP64 MFMA validation
//
// This test performs a moderate-sized DGEMM and reports expected FLOP counts
// for validation against the fp64_monitor tool's GEMM analysis.
//
// Build: hipcc -o test_rocblas_gemm test_rocblas_gemm.cpp -lrocblas
// Run:   ./test_rocblas_gemm [M] [N] [K] [iterations]

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <vector>

#define HIP_CHECK(cmd) do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(error) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define ROCBLAS_CHECK(cmd) do { \
    rocblas_status status = cmd; \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS error: " << status \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

int main(int argc, char* argv[]) {
    // Default sizes - moderate GEMM that runs long enough to sample
    int M = 4096;
    int N = 4096;
    int K = 4096;
    int iterations = 10;
    int warmup = 2;
    
    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) K = std::atoi(argv[3]);
    if (argc > 4) iterations = std::atoi(argv[4]);
    
    std::cout << "########################################################\n";
    std::cout << "#        rocBLAS DGEMM FP64 MFMA Test                  #\n";
    std::cout << "########################################################\n\n";
    
    // Get device info
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "Device: " << props.name << "\n";
    std::cout << "  Compute Units: " << props.multiProcessorCount << "\n";
    std::cout << "  Clock Rate:    " << props.clockRate / 1000 << " MHz\n\n";
    
    std::cout << "DGEMM Configuration:\n";
    std::cout << "  M = " << M << ", N = " << N << ", K = " << K << "\n";
    std::cout << "  Iterations: " << iterations << " (+ " << warmup << " warmup)\n";
    std::cout << "  Operation: C = alpha * A * B + beta * C\n";
    std::cout << "  Precision: FP64 (double)\n\n";
    
    // Calculate expected FLOPs
    // DGEMM: C[M,N] = A[M,K] * B[K,N]
    // Each element of C requires K multiply-adds = 2K FLOPs
    // Total = 2 * M * N * K FLOPs per GEMM
    double flops_per_gemm = 2.0 * M * N * K;
    double total_flops = flops_per_gemm * iterations;
    
    std::cout << "Expected FLOPs:\n";
    std::cout << "  Per GEMM:      " << std::scientific << std::setprecision(3) 
              << flops_per_gemm << " FLOPs\n";
    std::cout << "  Total (" << iterations << " iter): " << total_flops << " FLOPs\n";
    std::cout << "  Total:         " << std::fixed << std::setprecision(3) 
              << total_flops / 1e12 << " TFLOPs\n\n";
    
    // Allocate matrices
    size_t size_A = (size_t)M * K * sizeof(double);
    size_t size_B = (size_t)K * N * sizeof(double);
    size_t size_C = (size_t)M * N * sizeof(double);
    
    std::cout << "Memory:\n";
    std::cout << "  A: " << size_A / (1024.0 * 1024.0) << " MB\n";
    std::cout << "  B: " << size_B / (1024.0 * 1024.0) << " MB\n";
    std::cout << "  C: " << size_C / (1024.0 * 1024.0) << " MB\n";
    std::cout << "  Total: " << (size_A + size_B + size_C) / (1024.0 * 1024.0) << " MB\n\n";
    
    double *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));
    
    // Initialize with simple values
    std::vector<double> h_A(M * K, 1.0);
    std::vector<double> h_B(K * N, 1.0);
    std::vector<double> h_C(M * N, 0.0);
    
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice));
    
    // Create rocBLAS handle
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    
    double alpha = 1.0;
    double beta = 0.0;
    
    // Warmup
    std::cout << "Running " << warmup << " warmup iterations...\n";
    for (int i = 0; i < warmup; i++) {
        ROCBLAS_CHECK(rocblas_dgemm(handle,
                                     rocblas_operation_none,
                                     rocblas_operation_none,
                                     M, N, K,
                                     &alpha,
                                     d_A, M,
                                     d_B, K,
                                     &beta,
                                     d_C, M));
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Timed iterations
    std::cout << "Running " << iterations << " timed iterations...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        ROCBLAS_CHECK(rocblas_dgemm(handle,
                                     rocblas_operation_none,
                                     rocblas_operation_none,
                                     M, N, K,
                                     &alpha,
                                     d_A, M,
                                     d_B, K,
                                     &beta,
                                     d_C, M));
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // Calculate achieved performance
    double achieved_tflops = total_flops / elapsed.count() / 1e12;
    
    std::cout << "======================================================================\n";
    std::cout << "Results\n";
    std::cout << "======================================================================\n";
    std::cout << "  Time:          " << std::fixed << std::setprecision(3) 
              << elapsed.count() * 1000 << " ms\n";
    std::cout << "  Achieved:      " << std::setprecision(2) 
              << achieved_tflops << " TFLOPS\n\n";
    
    std::cout << "Expected vs Tool Comparison:\n";
    std::cout << "  The fp64_monitor tool should report approximately:\n";
    std::cout << "  - MFMA instructions (FP64 MFMA uses 4x4x4 = 128 FLOPs/inst)\n";
    std::cout << "  - Expected MFMA count: ~" << std::scientific 
              << total_flops / 128.0 << " instructions\n";
    std::cout << "  - GEMM Likelihood: Very High (>50% MFMA)\n";
    std::cout << "  - Estimated TFLOPS should be close to: " << std::fixed 
              << std::setprecision(2) << achieved_tflops << " TFLOPS\n\n";
    
    // Verify result (spot check)
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, size_C, hipMemcpyDeviceToHost));
    double expected_val = (double)K;  // Each element should be K (sum of 1*1 K times)
    double actual_val = h_C[0];
    bool pass = std::abs(actual_val - expected_val) < 1e-6;
    
    std::cout << "Verification:\n";
    std::cout << "  Expected C[0,0]: " << expected_val << "\n";
    std::cout << "  Actual C[0,0]:   " << actual_val << "\n";
    std::cout << "  Status:          " << (pass ? "PASS" : "FAIL") << "\n\n";
    
    std::cout << "########################################################\n";
    std::cout << "#                    Test Complete                      #\n";
    std::cout << "########################################################\n";
    
    // Cleanup
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    
    return pass ? 0 : 1;
}
