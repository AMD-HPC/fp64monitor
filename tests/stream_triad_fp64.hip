// STREAM Triad FP64 Test Program
// 
// This program performs the STREAM Triad operation on FP64 (double) arrays:
//   a[i] = b[i] + scalar * c[i]
//
// Expected FP64 instruction count per element:
//   - If compiler uses FMA: 1 FP64 FMA instruction per element
//   - If compiler uses separate ops: 1 FP64 MUL + 1 FP64 ADD per element
//
// For N elements with K iterations:
//   - FMA case: N * K FP64 FMA instructions
//   - Separate case: N * K FP64 MUL + N * K FP64 ADD instructions
//
// Build:
//   hipcc -O3 -o stream_triad_fp64 stream_triad_fp64.hip
//
// Usage:
//   ./stream_triad_fp64 [array_size] [iterations]
//   Default: array_size=1048576 (1M elements), iterations=100

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << hipGetErrorString(err) << std::endl;                   \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// STREAM Triad kernel: a[i] = b[i] + scalar * c[i]
// Each thread processes one element
// FP64 operations per element: 1 FMA (or 1 MUL + 1 ADD)
__global__ void stream_triad_kernel(double* __restrict__ a,
                                    const double* __restrict__ b,
                                    const double* __restrict__ c,
                                    double scalar,
                                    size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = b[idx] + scalar * c[idx];
    }
}

// Alternative kernel that explicitly uses FMA intrinsic
__global__ void stream_triad_fma_kernel(double* __restrict__ a,
                                        const double* __restrict__ b,
                                        const double* __restrict__ c,
                                        double scalar,
                                        size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // fma(x, y, z) = x * y + z
        a[idx] = fma(scalar, c[idx], b[idx]);
    }
}

// Kernel with explicit separate MUL and ADD (to prevent FMA fusion)
__global__ void stream_triad_separate_kernel(double* __restrict__ a,
                                             const double* __restrict__ b,
                                             const double* __restrict__ c,
                                             double scalar,
                                             size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use volatile to try to prevent FMA fusion
        volatile double temp = scalar * c[idx];
        a[idx] = b[idx] + temp;
    }
}

// Simple addition kernel for comparison (pure FP64 ADD)
__global__ void vector_add_kernel(double* __restrict__ a,
                                  const double* __restrict__ b,
                                  const double* __restrict__ c,
                                  size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = b[idx] + c[idx];
    }
}

// Simple multiply kernel for comparison (pure FP64 MUL)
__global__ void vector_mul_kernel(double* __restrict__ a,
                                  const double* __restrict__ b,
                                  const double* __restrict__ c,
                                  size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = b[idx] * c[idx];
    }
}

void print_expected_counts(const char* kernel_name, size_t n, int iterations,
                           size_t expected_add, size_t expected_mul, size_t expected_fma,
                           int warmup_iters = 5) {
    // SQ_INSTS counters measure per-wavefront (64 threads), not per-thread
    constexpr size_t WAVEFRONT_SIZE = 64;
    size_t total_launches = iterations + warmup_iters;
    size_t wavefront_insts_per_launch = n / WAVEFRONT_SIZE;
    
    // Counter values are per-wavefront
    size_t counter_add = (expected_add > 0) ? wavefront_insts_per_launch * total_launches : 0;
    size_t counter_mul = (expected_mul > 0) ? wavefront_insts_per_launch * total_launches : 0;
    size_t counter_fma = (expected_fma > 0) ? wavefront_insts_per_launch * total_launches : 0;
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Kernel: " << kernel_name << "\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Array size:      " << std::setw(15) << n << " elements\n";
    std::cout << "Iterations:      " << std::setw(15) << iterations << " (+ " << warmup_iters << " warmup)\n";
    std::cout << "Wavefronts/iter: " << std::setw(15) << wavefront_insts_per_launch << " (elements/64)\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Expected counter values (per-wavefront, cumulative):\n";
    std::cout << "  SQ_INSTS_VALU_ADD_F64: " << std::setw(15) << counter_add << "\n";
    std::cout << "  SQ_INSTS_VALU_MUL_F64: " << std::setw(15) << counter_mul << "\n";
    std::cout << "  SQ_INSTS_VALU_FMA_F64: " << std::setw(15) << counter_fma << "\n";
    std::cout << "  Total FP64 VALU:       " << std::setw(15) << (counter_add + counter_mul + counter_fma) << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void run_kernel(const char* name, int kernel_type, 
                double* d_a, double* d_b, double* d_c, 
                double scalar, size_t n, int iterations,
                int blocks, int threads) {
    
    constexpr int WARMUP_ITERS = 5;
    
    // Calculate expected FP64 ops per element (1 for single op kernels)
    size_t expected_add = 0, expected_mul = 0, expected_fma = 0;
    
    switch (kernel_type) {
        case 0: // Standard triad (likely FMA)
            expected_fma = 1;
            break;
        case 1: // Explicit FMA
            expected_fma = 1;
            break;
        case 2: // Separate MUL+ADD
            expected_add = 1;
            expected_mul = 1;
            break;
        case 3: // Vector add only
            expected_add = 1;
            break;
        case 4: // Vector mul only
            expected_mul = 1;
            break;
    }
    
    print_expected_counts(name, n, iterations, expected_add, expected_mul, expected_fma, WARMUP_ITERS);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        switch (kernel_type) {
            case 0:
                stream_triad_kernel<<<blocks, threads>>>(d_a, d_b, d_c, scalar, n);
                break;
            case 1:
                stream_triad_fma_kernel<<<blocks, threads>>>(d_a, d_b, d_c, scalar, n);
                break;
            case 2:
                stream_triad_separate_kernel<<<blocks, threads>>>(d_a, d_b, d_c, scalar, n);
                break;
            case 3:
                vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
                break;
            case 4:
                vector_mul_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
                break;
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    std::cout << "\nRunning " << iterations << " iterations...\n";
    
    // Small delay to allow profiler to start
    HIP_CHECK(hipDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Timed iterations
    for (int i = 0; i < iterations; i++) {
        switch (kernel_type) {
            case 0:
                stream_triad_kernel<<<blocks, threads>>>(d_a, d_b, d_c, scalar, n);
                break;
            case 1:
                stream_triad_fma_kernel<<<blocks, threads>>>(d_a, d_b, d_c, scalar, n);
                break;
            case 2:
                stream_triad_separate_kernel<<<blocks, threads>>>(d_a, d_b, d_c, scalar, n);
                break;
            case 3:
                vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
                break;
            case 4:
                vector_mul_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
                break;
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double bandwidth_gb = (3.0 * n * sizeof(double) * iterations) / (elapsed_ms * 1e6);
    size_t total_elements = (size_t)n * iterations;
    double gflops = (double)total_elements / (elapsed_ms * 1e6);
    if (kernel_type == 0 || kernel_type == 1) {
        gflops *= 2.0; // FMA = 2 FLOPs
    } else if (kernel_type == 2) {
        gflops *= 2.0; // MUL + ADD = 2 FLOPs
    }
    
    std::cout << "\nPerformance:\n";
    std::cout << "  Time:      " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n";
    std::cout << "  Bandwidth: " << bandwidth_gb << " GB/s\n";
    std::cout << "  GFLOP/s:   " << gflops << "\n";
}

int main(int argc, char* argv[]) {
    // Default parameters
    size_t n = 1024 * 1024;  // 1M elements = 8 MB per array
    int iterations = 100;
    int kernel_type = -1;    // -1 = run all kernels
    
    // Parse command line arguments
    if (argc > 1) {
        n = std::stoull(argv[1]);
    }
    if (argc > 2) {
        iterations = std::stoi(argv[2]);
    }
    if (argc > 3) {
        kernel_type = std::stoi(argv[3]);
    }
    
    std::cout << "\n";
    std::cout << "########################################################\n";
    std::cout << "#          STREAM Triad FP64 Test Program              #\n";
    std::cout << "########################################################\n";
    std::cout << "\n";
    std::cout << "This program generates known FP64 instruction counts\n";
    std::cout << "for validating hardware counter collection.\n\n";
    
    // Print device info
    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    
    std::cout << "Device: " << props.name << "\n";
    std::cout << "  Compute Units: " << props.multiProcessorCount << "\n";
    std::cout << "  Clock Rate:    " << props.clockRate / 1000 << " MHz\n";
    std::cout << "  Memory:        " << props.totalGlobalMem / (1024*1024*1024) << " GB\n";
    std::cout << "\n";
    
    // Allocate arrays
    double *h_a, *h_b, *h_c;
    double *d_a, *d_b, *d_c;
    
    size_t bytes = n * sizeof(double);
    
    h_a = new double[n];
    h_b = new double[n];
    h_c = new double[n];
    
    // Initialize host arrays
    for (size_t i = 0; i < n; i++) {
        h_a[i] = 0.0;
        h_b[i] = 1.0;
        h_c[i] = 2.0;
    }
    double scalar = 3.0;
    
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_c, h_c, bytes, hipMemcpyHostToDevice));
    
    // Calculate grid dimensions
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    std::cout << "Configuration:\n";
    std::cout << "  Array size:   " << n << " elements (" << bytes / (1024*1024) << " MB per array)\n";
    std::cout << "  Iterations:   " << iterations << "\n";
    std::cout << "  Grid:         " << blocks << " blocks x " << threads << " threads\n";
    std::cout << "  Scalar value: " << scalar << "\n";
    
    // Run selected kernel(s)
    if (kernel_type == -1 || kernel_type == 0) {
        run_kernel("STREAM Triad (standard)", 0, d_a, d_b, d_c, scalar, n, iterations, blocks, threads);
    }
    if (kernel_type == -1 || kernel_type == 1) {
        run_kernel("STREAM Triad (explicit FMA)", 1, d_a, d_b, d_c, scalar, n, iterations, blocks, threads);
    }
    if (kernel_type == -1 || kernel_type == 2) {
        run_kernel("STREAM Triad (separate MUL+ADD)", 2, d_a, d_b, d_c, scalar, n, iterations, blocks, threads);
    }
    if (kernel_type == -1 || kernel_type == 3) {
        run_kernel("Vector Add (ADD only)", 3, d_a, d_b, d_c, scalar, n, iterations, blocks, threads);
    }
    if (kernel_type == -1 || kernel_type == 4) {
        run_kernel("Vector Multiply (MUL only)", 4, d_a, d_b, d_c, scalar, n, iterations, blocks, threads);
    }
    
    // Verify results
    HIP_CHECK(hipMemcpy(h_a, d_a, bytes, hipMemcpyDeviceToHost));
    
    double expected;
    switch (kernel_type == -1 ? 4 : kernel_type) {  // Check last kernel run
        case 0:
        case 1:
        case 2:
            expected = h_b[0] + scalar * h_c[0];  // 1.0 + 3.0 * 2.0 = 7.0
            break;
        case 3:
            expected = h_b[0] + h_c[0];  // 1.0 + 2.0 = 3.0
            break;
        case 4:
            expected = h_b[0] * h_c[0];  // 1.0 * 2.0 = 2.0
            break;
        default:
            expected = 0.0;
    }
    
    std::cout << "\nVerification:\n";
    std::cout << "  Expected a[0]: " << expected << "\n";
    std::cout << "  Actual a[0]:   " << h_a[0] << "\n";
    std::cout << "  Status:        " << (std::fabs(h_a[0] - expected) < 1e-10 ? "PASS" : "FAIL") << "\n";
    
    // Cleanup
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    std::cout << "\n########################################################\n";
    std::cout << "#                    Test Complete                      #\n";
    std::cout << "########################################################\n\n";
    
    return 0;
}
