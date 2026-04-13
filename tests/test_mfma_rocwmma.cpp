// rocWMMA MFMA test - uses rocWMMA library for FP64 matrix multiply-accumulate
// This validates the fp64_monitor tool's GEMM analysis feature using the
// high-level rocWMMA API instead of inline assembly.
//
// Note: Requires rocWMMA headers (part of ROCm)
//
// Build: hipcc -I$ROCM_PATH/include -o test_mfma_rocwmma test_mfma_rocwmma.cpp
// Run:   ./test_mfma_rocwmma [iterations] [kernel_launches]
//
// Expected validation:
//   - FP64 MFMA counter should match expected MFMA count
//   - TFLOPS should be approximately equal to "Achieved" output
//   - GEMM Likelihood should be "Very High"

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

using namespace rocwmma;

#define HIP_CHECK(cmd) do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(error) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// rocWMMA tile dimensions for FP64
// Supported: 16x16x4 on gfx90a, gfx942, gfx950
constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;
constexpr int BLOCK_K = 4;

// Matrix dimensions (must be multiples of block dimensions)
constexpr int M = 256;
constexpr int N = 256;
constexpr int K = 256;

// Kernel that executes FP64 MFMA instructions using rocWMMA
// Each wavefront computes one 16x16 output tile
__global__ void mfma_fp64_rocwmma_kernel(double* D, const double* A, const double* B,
                                          const double* C, int iterations) {
    // Fragment declarations for FP64 GEMM
    // A: M x K tile, B: K x N tile, C/D: M x N accumulator
    // Note: accumulator also needs layout specified for load/store
    fragment<matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, double, row_major> fragA;
    fragment<matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, double, col_major> fragB;
    fragment<accumulator, BLOCK_M, BLOCK_N, BLOCK_K, double, row_major> fragC;

    // Calculate tile position
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    // Pointers to this tile's data
    const double* tileA = A + tileRow * BLOCK_M * K;
    const double* tileB = B + tileCol * BLOCK_N;
    double* tileD = D + tileRow * BLOCK_M * N + tileCol * BLOCK_N;
    const double* tileC = C + tileRow * BLOCK_M * N + tileCol * BLOCK_N;

    // Load initial accumulator (C matrix)
    load_matrix_sync(fragC, tileC, N);

    // Perform iterations of MFMA to generate measurable workload
    for (int iter = 0; iter < iterations; iter++) {
        // Walk along K dimension
        for (int k = 0; k < K; k += BLOCK_K) {
            // Load A tile (BLOCK_M x BLOCK_K)
            load_matrix_sync(fragA, tileA + k, K);

            // Load B tile (BLOCK_K x BLOCK_N)
            load_matrix_sync(fragB, tileB + k * N, N);

            // Multiply-accumulate: D = A * B + C
            mma_sync(fragC, fragA, fragB, fragC);
        }
    }

    // Store result
    store_matrix_sync(tileD, fragC, N);
}

// Simple kernel for comparison - just fills with zeros
__global__ void init_matrix_kernel(double* mat, int rows, int cols, double val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        mat[idx] = val;
    }
}

int main(int argc, char* argv[]) {
    int iterations = 100;       // GEMM iterations per kernel
    int kernel_launches = 10;   // Number of kernel launches

    if (argc > 1) iterations = std::atoi(argv[1]);
    if (argc > 2) kernel_launches = std::atoi(argv[2]);

    std::cout << "########################################################\n";
    std::cout << "#         rocWMMA FP64 MFMA Test                       #\n";
    std::cout << "########################################################\n\n";

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "Device: " << props.name << "\n";
    std::cout << "  Compute Units: " << props.multiProcessorCount << "\n";
    std::cout << "  Wavefront Size: " << props.warpSize << "\n\n";

    // Check architecture support
    std::string gcnArch = props.gcnArchName;
    bool supported = (gcnArch.find("gfx90a") != std::string::npos ||
                      gcnArch.find("gfx942") != std::string::npos ||
                      gcnArch.find("gfx940") != std::string::npos ||
                      gcnArch.find("gfx941") != std::string::npos ||
                      gcnArch.find("gfx950") != std::string::npos);

    if (!supported) {
        std::cerr << "Warning: GPU architecture " << gcnArch
                  << " may not support FP64 MFMA instructions.\n";
        std::cerr << "Supported: gfx90a, gfx940, gfx941, gfx942, gfx950\n\n";
    }

    std::cout << "Configuration:\n";
    std::cout << "  Matrix size:     " << M << " x " << N << " x " << K << "\n";
    std::cout << "  Block size:      " << BLOCK_M << " x " << BLOCK_N << " x " << BLOCK_K << "\n";
    std::cout << "  Iterations:      " << iterations << " per kernel\n";
    std::cout << "  Kernel launches: " << kernel_launches << "\n\n";

    // Calculate grid dimensions
    int gridX = N / BLOCK_N;  // Tiles in X (columns)
    int gridY = M / BLOCK_M;  // Tiles in Y (rows)
    int wavefront_size = props.warpSize;

    std::cout << "  Grid:            " << gridX << " x " << gridY << " tiles\n";
    std::cout << "  Threads/block:   " << wavefront_size << " (1 wavefront)\n\n";

    // Calculate expected MFMA instructions and FLOPs
    // Each mma_sync on a 16x16x4 tile = one MFMA instruction per wavefront
    // K dimension is walked in steps of BLOCK_K, so K/BLOCK_K mma_sync calls per iteration
    long long mma_per_tile_per_iter = K / BLOCK_K;  // mma_sync calls per tile per iteration
    long long total_tiles = (long long)gridX * gridY;
    long long total_mma_per_launch = total_tiles * mma_per_tile_per_iter * iterations;
    long long total_mfma_counter = total_mma_per_launch * kernel_launches;

    // FLOPs: each 16x16x4 MFMA = 16 * 16 * 4 * 2 = 2048 FLOPs
    long long flops_per_mfma = BLOCK_M * BLOCK_N * BLOCK_K * 2;
    long long total_flops = total_mfma_counter * flops_per_mfma;

    std::cout << "Expected Hardware Counter Values:\n";
    std::cout << "  MMA ops/tile/iter:  " << mma_per_tile_per_iter << "\n";
    std::cout << "  Total tiles:        " << total_tiles << "\n";
    std::cout << "  Total MFMA (counter): " << std::scientific << std::setprecision(3)
              << (double)total_mfma_counter << "\n";
    std::cout << "  FLOPs/MFMA:         " << flops_per_mfma << "\n";
    std::cout << "  Total FLOPs:        " << (double)total_flops << "\n";
    std::cout << "  Total TFLOPs:       " << std::fixed << std::setprecision(3)
              << total_flops / 1e12 << "\n\n";

    // Allocate matrices
    double *d_A, *d_B, *d_C, *d_D;
    size_t size_A = M * K * sizeof(double);
    size_t size_B = K * N * sizeof(double);
    size_t size_C = M * N * sizeof(double);
    size_t size_D = M * N * sizeof(double);

    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));
    HIP_CHECK(hipMalloc(&d_D, size_D));

    // Initialize matrices with simple values
    int init_threads = 256;
    int init_blocks_A = (M * K + init_threads - 1) / init_threads;
    int init_blocks_B = (K * N + init_threads - 1) / init_threads;
    int init_blocks_C = (M * N + init_threads - 1) / init_threads;

    init_matrix_kernel<<<init_blocks_A, init_threads>>>(d_A, M, K, 1.0);
    init_matrix_kernel<<<init_blocks_B, init_threads>>>(d_B, K, N, 1.0);
    init_matrix_kernel<<<init_blocks_C, init_threads>>>(d_C, M, N, 0.0);
    HIP_CHECK(hipDeviceSynchronize());

    // Kernel launch configuration
    dim3 grid(gridX, gridY);
    dim3 block(wavefront_size);  // One wavefront per block for simplicity

    // Warmup
    std::cout << "Running warmup...\n";
    mfma_fp64_rocwmma_kernel<<<grid, block>>>(d_D, d_A, d_B, d_C, iterations);
    HIP_CHECK(hipDeviceSynchronize());

    // Reset C for timed run
    init_matrix_kernel<<<init_blocks_C, init_threads>>>(d_C, M, N, 0.0);
    HIP_CHECK(hipDeviceSynchronize());

    // Timed run
    std::cout << "Running " << kernel_launches << " kernel launches...\n";

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kernel_launches; i++) {
        mfma_fp64_rocwmma_kernel<<<grid, block>>>(d_D, d_A, d_B, d_C, iterations);
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
    HIP_CHECK(hipFree(d_D));

    return 0;
}
