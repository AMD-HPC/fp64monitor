/**
 * SpMV Test Program for Counter Validation
 * 
 * Sparse Matrix-Vector multiplication (y = A * x) using CSR format.
 * This test exercises:
 *   - FP64 operations (double precision SpMV)
 *   - FP32 operations (single precision SpMV)
 *   - Integer operations (index calculations, array accesses)
 * 
 * Usage:
 *   ./test_spmv [--fp64|--fp32|--int] [--rows N] [--nnz-per-row M] [--iterations I]
 * 
 * For counter validation, run with the appropriate counter set:
 *   FP64_MONITOR_COUNTERS=fp64 ./test_spmv --fp64
 *   FP64_MONITOR_COUNTERS=fp32 ./test_spmv --fp32
 *   FP64_MONITOR_COUNTERS=int  ./test_spmv --int
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cstdint>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// CSR SpMV kernel - FP64
__global__ void spmv_csr_fp64(
    const int num_rows,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        double sum = 0.0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}

// CSR SpMV kernel - FP32
__global__ void spmv_csr_fp32(
    const int num_rows,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}

// Integer-heavy kernel: histogram/counting with minimal FP
// Each thread processes multiple elements and accumulates integer counters
__global__ void integer_workload(
    const int n,
    const int* __restrict__ input,
    int* __restrict__ histogram,
    const int num_bins)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n; i += stride) {
        int val = input[i];
        int bin = val % num_bins;
        atomicAdd(&histogram[bin], 1);
    }
}

// Separate integer kernel that does index-heavy computation
__global__ void index_computation_kernel(
    const int n,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int* __restrict__ output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int start = row_ptr[tid];
        int end = row_ptr[tid + 1];
        int count = end - start;
        int sum = 0;
        for (int j = start; j < end; j++) {
            sum += col_idx[j];
        }
        output[tid] = (count > 0) ? (sum / count) : 0;
    }
}

void print_header() {
    std::cout << "\n########################################################\n";
    std::cout << "#          SpMV Test Program for Counter Validation    #\n";
    std::cout << "########################################################\n\n";
}

void print_device_info() {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    
    std::cout << "Device: " << props.name << "\n";
    std::cout << "  Compute Units: " << props.multiProcessorCount << "\n";
    std::cout << "  Clock Rate:    " << props.clockRate / 1000 << " MHz\n";
    std::cout << "  Memory:        " << props.totalGlobalMem / (1024*1024*1024) << " GB\n\n";
}

void run_fp64_test(int num_rows, int nnz_per_row, int iterations) {
    std::cout << "======================================================================\n";
    std::cout << "SpMV FP64 Test\n";
    std::cout << "----------------------------------------------------------------------\n";
    
    int num_cols = num_rows;
    int64_t nnz = (int64_t)num_rows * nnz_per_row;
    
    std::cout << "Matrix:      " << num_rows << " x " << num_cols << "\n";
    std::cout << "NNZ:         " << nnz << " (" << nnz_per_row << " per row)\n";
    std::cout << "Iterations:  " << iterations << "\n\n";
    
    // Allocate and initialize host data
    std::vector<int> h_row_ptr(num_rows + 1);
    std::vector<int> h_col_idx(nnz);
    std::vector<double> h_values(nnz);
    std::vector<double> h_x(num_cols, 1.0);
    std::vector<double> h_y(num_rows, 0.0);
    
    // Generate CSR structure (regular pattern for predictable counters)
    std::mt19937 rng(42);
    for (int i = 0; i <= num_rows; i++) {
        h_row_ptr[i] = i * nnz_per_row;
    }
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < nnz_per_row; j++) {
            h_col_idx[i * nnz_per_row + j] = (i + j) % num_cols;
            h_values[i * nnz_per_row + j] = 1.0;
        }
    }
    
    // Allocate device memory
    int *d_row_ptr, *d_col_idx;
    double *d_values, *d_x, *d_y;
    
    HIP_CHECK(hipMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_idx, nnz * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_values, nnz * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x, num_cols * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_y, num_rows * sizeof(double)));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_row_ptr, h_row_ptr.data(), (num_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_values, h_values.data(), nnz * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), num_cols * sizeof(double), hipMemcpyHostToDevice));
    
    // Launch config
    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;
    int wavefronts_per_launch = (num_rows + 63) / 64;
    
    // Expected counters per iteration:
    // Each thread processes one row with nnz_per_row elements
    // Each element: 1 FMA (or MUL+ADD)
    // Per wavefront (64 threads): 64 * nnz_per_row FMAs
    int64_t fp64_fma_per_iter = (int64_t)wavefronts_per_launch * 64 * nnz_per_row;
    // But we only have num_rows active threads
    fp64_fma_per_iter = (int64_t)num_rows * nnz_per_row;
    
    std::cout << "Expected FP64 FMA instructions: " << fp64_fma_per_iter * iterations << "\n";
    std::cout << "  (per iteration: " << fp64_fma_per_iter << ")\n\n";
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        spmv_csr_fp64<<<num_blocks, block_size>>>(num_rows, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Timed iterations
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        spmv_csr_fp64<<<num_blocks, block_size>>>(num_rows, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    
    double gflops = (2.0 * nnz * iterations) / (ms * 1e6);
    double gbps = ((nnz * (sizeof(double) + sizeof(int)) + num_rows * sizeof(int) + num_cols * sizeof(double)) * iterations) / (ms * 1e6);
    
    std::cout << "Performance:\n";
    std::cout << "  Time:      " << ms << " ms\n";
    std::cout << "  GFLOP/s:   " << gflops << "\n";
    std::cout << "  GB/s:      " << gbps << "\n\n";
    
    // Verify
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, num_rows * sizeof(double), hipMemcpyDeviceToHost));
    double expected = nnz_per_row; // Each row sums nnz_per_row values of 1.0*1.0
    bool pass = (std::abs(h_y[0] - expected) < 1e-6);
    std::cout << "Verification: y[0] = " << h_y[0] << " (expected " << expected << ") - " 
              << (pass ? "PASS" : "FAIL") << "\n\n";
    
    // Cleanup
    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_col_idx));
    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

void run_fp32_test(int num_rows, int nnz_per_row, int iterations) {
    std::cout << "======================================================================\n";
    std::cout << "SpMV FP32 Test\n";
    std::cout << "----------------------------------------------------------------------\n";
    
    int num_cols = num_rows;
    int64_t nnz = (int64_t)num_rows * nnz_per_row;
    
    std::cout << "Matrix:      " << num_rows << " x " << num_cols << "\n";
    std::cout << "NNZ:         " << nnz << " (" << nnz_per_row << " per row)\n";
    std::cout << "Iterations:  " << iterations << "\n\n";
    
    // Allocate and initialize host data
    std::vector<int> h_row_ptr(num_rows + 1);
    std::vector<int> h_col_idx(nnz);
    std::vector<float> h_values(nnz);
    std::vector<float> h_x(num_cols, 1.0f);
    std::vector<float> h_y(num_rows, 0.0f);
    
    // Generate CSR structure
    for (int i = 0; i <= num_rows; i++) {
        h_row_ptr[i] = i * nnz_per_row;
    }
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < nnz_per_row; j++) {
            h_col_idx[i * nnz_per_row + j] = (i + j) % num_cols;
            h_values[i * nnz_per_row + j] = 1.0f;
        }
    }
    
    // Allocate device memory
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;
    
    HIP_CHECK(hipMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_idx, nnz * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_values, nnz * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x, num_cols * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, num_rows * sizeof(float)));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_row_ptr, h_row_ptr.data(), (num_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_values, h_values.data(), nnz * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), num_cols * sizeof(float), hipMemcpyHostToDevice));
    
    // Launch config
    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;
    
    int64_t fp32_fma_per_iter = (int64_t)num_rows * nnz_per_row;
    
    std::cout << "Expected FP32 FMA instructions: " << fp32_fma_per_iter * iterations << "\n";
    std::cout << "  (per iteration: " << fp32_fma_per_iter << ")\n\n";
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        spmv_csr_fp32<<<num_blocks, block_size>>>(num_rows, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Timed iterations
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        spmv_csr_fp32<<<num_blocks, block_size>>>(num_rows, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    
    double gflops = (2.0 * nnz * iterations) / (ms * 1e6);
    
    std::cout << "Performance:\n";
    std::cout << "  Time:      " << ms << " ms\n";
    std::cout << "  GFLOP/s:   " << gflops << "\n\n";
    
    // Verify
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, num_rows * sizeof(float), hipMemcpyDeviceToHost));
    float expected = (float)nnz_per_row;
    bool pass = (std::abs(h_y[0] - expected) < 1e-4f);
    std::cout << "Verification: y[0] = " << h_y[0] << " (expected " << expected << ") - " 
              << (pass ? "PASS" : "FAIL") << "\n\n";
    
    // Cleanup
    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_col_idx));
    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

void run_int_test(int num_rows, int nnz_per_row, int iterations) {
    std::cout << "======================================================================\n";
    std::cout << "Integer Workload Test\n";
    std::cout << "----------------------------------------------------------------------\n";
    
    int64_t nnz = (int64_t)num_rows * nnz_per_row;
    
    std::cout << "Rows:        " << num_rows << "\n";
    std::cout << "NNZ:         " << nnz << " (" << nnz_per_row << " per row)\n";
    std::cout << "Iterations:  " << iterations << "\n\n";
    
    // Allocate and initialize host data
    std::vector<int> h_row_ptr(num_rows + 1);
    std::vector<int> h_col_idx(nnz);
    std::vector<int> h_output(num_rows, 0);
    
    // Generate CSR structure
    for (int i = 0; i <= num_rows; i++) {
        h_row_ptr[i] = i * nnz_per_row;
    }
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < nnz_per_row; j++) {
            h_col_idx[i * nnz_per_row + j] = (i + j) % num_rows;
        }
    }
    
    // Allocate device memory
    int *d_row_ptr, *d_col_idx, *d_output;
    
    HIP_CHECK(hipMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_idx, nnz * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_output, num_rows * sizeof(int)));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_row_ptr, h_row_ptr.data(), (num_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), hipMemcpyHostToDevice));
    
    // Launch config
    int block_size = 256;
    int num_blocks = (num_rows + block_size - 1) / block_size;
    
    // Expected integer operations per iteration:
    // Per row: 2 loads (row_ptr[row], row_ptr[row+1])
    //          1 subtraction (count = end - start)
    //          nnz_per_row loads (col_idx[j])
    //          nnz_per_row additions (sum += col_idx[j])
    //          1 division (sum / count)
    // Approximate INT32 ops per thread: ~2 + nnz_per_row
    int64_t int32_ops_per_iter = (int64_t)num_rows * (2 + nnz_per_row);
    
    std::cout << "Expected INT32 operations (approx): " << int32_ops_per_iter * iterations << "\n";
    std::cout << "  (per iteration: " << int32_ops_per_iter << ")\n\n";
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        index_computation_kernel<<<num_blocks, block_size>>>(num_rows, d_row_ptr, d_col_idx, d_output);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Timed iterations
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        index_computation_kernel<<<num_blocks, block_size>>>(num_rows, d_row_ptr, d_col_idx, d_output);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    
    std::cout << "Performance:\n";
    std::cout << "  Time:      " << ms << " ms\n\n";
    
    // Verify
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, num_rows * sizeof(int), hipMemcpyDeviceToHost));
    // Row 0: indices are 0,1,2,...,nnz_per_row-1, average = (nnz_per_row-1)/2
    int expected = (nnz_per_row - 1) / 2;
    bool pass = (h_output[0] == expected);
    std::cout << "Verification: output[0] = " << h_output[0] << " (expected " << expected << ") - " 
              << (pass ? "PASS" : "FAIL") << "\n\n";
    
    // Cleanup
    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_col_idx));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --fp64          Run FP64 SpMV test (default)\n";
    std::cout << "  --fp32          Run FP32 SpMV test\n";
    std::cout << "  --int           Run integer workload test\n";
    std::cout << "  --all           Run all tests\n";
    std::cout << "  --rows N        Number of matrix rows (default: 1048576)\n";
    std::cout << "  --nnz-per-row M Non-zeros per row (default: 16)\n";
    std::cout << "  --iterations I  Number of iterations (default: 100)\n";
    std::cout << "  --help          Show this help\n\n";
    std::cout << "For counter validation, set FP64_MONITOR_COUNTERS appropriately:\n";
    std::cout << "  FP64_MONITOR_COUNTERS=fp64 ./test_spmv --fp64\n";
    std::cout << "  FP64_MONITOR_COUNTERS=fp32 ./test_spmv --fp32\n";
    std::cout << "  FP64_MONITOR_COUNTERS=int  ./test_spmv --int\n";
}

int main(int argc, char** argv) {
    // Defaults
    int num_rows = 1048576;
    int nnz_per_row = 16;
    int iterations = 100;
    bool run_fp64 = false;
    bool run_fp32 = false;
    bool run_int = false;
    bool run_all = false;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fp64") == 0) {
            run_fp64 = true;
        } else if (strcmp(argv[i], "--fp32") == 0) {
            run_fp32 = true;
        } else if (strcmp(argv[i], "--int") == 0) {
            run_int = true;
        } else if (strcmp(argv[i], "--all") == 0) {
            run_all = true;
        } else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            num_rows = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nnz-per-row") == 0 && i + 1 < argc) {
            nnz_per_row = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Default to fp64 if nothing specified
    if (!run_fp64 && !run_fp32 && !run_int && !run_all) {
        run_fp64 = true;
    }
    
    if (run_all) {
        run_fp64 = run_fp32 = run_int = true;
    }
    
    print_header();
    print_device_info();
    
    std::cout << "Configuration:\n";
    std::cout << "  Rows:         " << num_rows << "\n";
    std::cout << "  NNZ per row:  " << nnz_per_row << "\n";
    std::cout << "  Iterations:   " << iterations << "\n\n";
    
    if (run_fp64) {
        run_fp64_test(num_rows, nnz_per_row, iterations);
    }
    
    if (run_fp32) {
        run_fp32_test(num_rows, nnz_per_row, iterations);
    }
    
    if (run_int) {
        run_int_test(num_rows, nnz_per_row, iterations);
    }
    
    std::cout << "########################################################\n";
    std::cout << "#                    Test Complete                      #\n";
    std::cout << "########################################################\n\n";
    
    return 0;
}
