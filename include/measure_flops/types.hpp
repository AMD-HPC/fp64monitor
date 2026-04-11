#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace measure_flops {

// Counter categories for instruction mix analysis
enum class CounterCategory {
    FP64_VALU,   // F64 vector ALU (ADD, MUL, FMA, TRANS)
    FP64_MFMA,   // F64 matrix FMA
    FP32_VALU,   // F32 vector ALU
    FP32_MFMA,   // F32 matrix FMA
    FP16_VALU,   // F16 vector ALU
    FP16_MFMA,   // F16 matrix FMA
    BF16_MFMA,   // BF16 matrix FMA
    INT8_MFMA,   // INT8 matrix FMA
    INT32,       // 32-bit integer
    INT64,       // 64-bit integer
    MEMORY,      // Vector memory operations
    SCALAR,      // Scalar ALU + memory
    LDS,         // Local data share
    BRANCH,      // Branch instructions
    OTHER,       // Everything else
    TOTAL        // Total instructions
};

// Single counter value with metadata
struct CounterValue {
    std::string name;
    double value;
    uint64_t counter_id;
    uint64_t instance_id;
    std::map<std::string, size_t> dimensions;  // XCC, SE, etc.
};

// Sample containing all counter values at a point in time
struct Sample {
    std::chrono::steady_clock::time_point timestamp;
    uint64_t sample_id;
    int gpu_id;
    std::vector<CounterValue> counters;
    
    Sample() : sample_id(0), gpu_id(-1) {}
};

// Analyzed instruction mix from a sample
struct InstructionMix {
    std::chrono::steady_clock::time_point timestamp;
    uint64_t sample_id;
    int gpu_id;

    // Raw instruction counts (summed across all instances)
    double total_instructions;
    double valu_instructions;

    // FP64 breakdown
    double fp64_add;
    double fp64_mul;
    double fp64_fma;
    double fp64_trans;
    double fp64_mfma;
    double fp64_mfma_ops;  // Actual FLOP count (MFMA_MOPS * 512)

    // Other precision types
    double fp32_add;
    double fp32_mul;
    double fp32_fma;
    double fp32_trans;
    double fp32_mfma;
    double fp16_total;
    double bf16_mfma;
    double int8_mfma;
    double int32_total;
    double int64_total;

    // Memory and other
    double vmem_total;
    double vmem_rd;
    double vmem_wr;
    double vmem_rd_cycles;
    double vmem_wr_cycles;
    double smem_total;
    double lds_total;
    double salu_total;
    double branch_total;
    double mfma_total;

    // Computed percentages
    double fp64_valu_pct;     // FP64 VALU as % of total VALU
    double fp64_total_pct;    // All FP64 as % of total instructions
    double fp64_compute_pct;  // FP64 as % of compute (excluding mem/branch)
    double mfma_fp64_pct;     // FP64 MFMA as % of all MFMA

    // Memory statistics (bytes)
    uint64_t gpu_mem_used{0};      // GPU memory used (from HIP)
    uint64_t gpu_mem_total{0};     // GPU memory total (from HIP)
    uint64_t vram_used{0};         // VRAM used (from ROCm SMI)
    uint64_t vram_total{0};        // VRAM total (from ROCm SMI)
    uint64_t gtt_used{0};          // GTT used - system memory for GPU (from ROCm SMI)
    uint64_t gtt_total{0};         // GTT total (from ROCm SMI)
    uint64_t cpu_mem_resident{0};  // CPU resident memory (from /proc/self/statm)
    uint64_t cpu_mem_virtual{0};   // CPU virtual memory (from /proc/self/statm)
    
    // High water marks
    uint64_t gpu_mem_hwm{0};       // Peak GPU memory
    uint64_t vram_hwm{0};          // Peak VRAM
    uint64_t gtt_hwm{0};           // Peak GTT
    uint64_t cpu_mem_hwm{0};       // Peak CPU resident memory
    uint64_t cpu_vm_hwm{0};        // Peak CPU virtual memory

    InstructionMix()
        : sample_id(0),
          gpu_id(-1),
          total_instructions(0),
          valu_instructions(0),
          fp64_add(0),
          fp64_mul(0),
          fp64_fma(0),
          fp64_trans(0),
          fp64_mfma(0),
          fp64_mfma_ops(0),
          fp32_add(0),
          fp32_mul(0),
          fp32_fma(0),
          fp32_trans(0),
          fp32_mfma(0),
          fp16_total(0),
          bf16_mfma(0),
          int8_mfma(0),
          int32_total(0),
          int64_total(0),
          vmem_total(0),
          vmem_rd(0),
          vmem_wr(0),
          vmem_rd_cycles(0),
          vmem_wr_cycles(0),
          smem_total(0),
          lds_total(0),
          salu_total(0),
          branch_total(0),
          mfma_total(0),
          fp64_valu_pct(0),
          fp64_total_pct(0),
          fp64_compute_pct(0),
          mfma_fp64_pct(0) {}
};

// Configuration for the monitor
struct MonitorConfig {
    std::chrono::milliseconds sampling_interval{100};
    std::vector<int> gpu_ids;            // Empty = all GPUs
    std::string output_file;             // Empty = stdout
    std::string output_format{"text"};   // text, csv, json
    bool aggregate_dimensions{true};     // Sum across XCC/SE
    bool continuous{true};               // Run until interrupted
    int num_samples{0};                  // 0 = unlimited (if continuous)
    bool verbose{false};
    bool full_instruction_mix{false};    // Collect all counters vs FP64 only
    bool list_counters{false};           // List available counters and exit
    bool list_gpus{false};               // List available GPUs and exit
    bool use_rocm_smi{false};
    std::string tmpdir;                  // Fast local storage for temp output (e.g., /dev/shm)
};

// Predefined counter sets
struct CounterSets {
    // Minimal set for FP64 analysis only (8 counters - at hardware limit)
    static std::vector<std::string> fp64_minimal() {
        return {
            "SQ_INSTS",
            "SQ_INSTS_VALU",
            "SQ_INSTS_VALU_ADD_F64",
            "SQ_INSTS_VALU_MUL_F64",
            "SQ_INSTS_VALU_FMA_F64",
            "SQ_INSTS_VALU_TRANS_F64",
            "SQ_INSTS_VALU_MFMA_F64",
            "SQ_INSTS_MFMA"
        };
    }

    // Extended set including F32 for comparison
    static std::vector<std::string> fp64_with_fp32() {
        return {
            "SQ_INSTS",
            "SQ_INSTS_VALU",
            "SQ_INSTS_SALU",
            "SQ_INSTS_VALU_ADD_F64",
            "SQ_INSTS_VALU_MUL_F64",
            "SQ_INSTS_VALU_FMA_F64",
            "SQ_INSTS_VALU_TRANS_F64",
            "SQ_INSTS_VALU_MFMA_F64",
            "SQ_INSTS_VALU_ADD_F32",
            "SQ_INSTS_VALU_MUL_F32",
            "SQ_INSTS_VALU_FMA_F32",
            "SQ_INSTS_VALU_TRANS_F32",
            "SQ_INSTS_VALU_MFMA_F32",
            "SQ_INSTS_MFMA"
        };
    }

    // FP32-focused counter set (8 counters - fits hardware limit)
    static std::vector<std::string> fp32_minimal() {
        return {
            "SQ_INSTS",
            "SQ_INSTS_VALU",
            "SQ_INSTS_MFMA",
            "SQ_INSTS_VALU_ADD_F32",
            "SQ_INSTS_VALU_MUL_F32",
            "SQ_INSTS_VALU_FMA_F32",
            "SQ_INSTS_VALU_TRANS_F32",
            "SQ_INSTS_VALU_MFMA_F32"
        };
    }

    // FP16/BF16-focused counter set (8 counters - fits hardware limit)
    static std::vector<std::string> fp16_minimal() {
        return {
            "SQ_INSTS",
            "SQ_INSTS_VALU",
            "SQ_INSTS_MFMA",
            "SQ_INSTS_VALU_ADD_F16",
            "SQ_INSTS_VALU_MUL_F16",
            "SQ_INSTS_VALU_FMA_F16",
            "SQ_INSTS_VALU_MFMA_F16",
            "SQ_INSTS_VALU_MFMA_BF16"
        };
    }

    // Integer + VMEM counter set (8 counters - at hardware limit)
    static std::vector<std::string> int_minimal() {
        return {
            "SQ_INSTS",
            "SQ_INSTS_VALU",
            "SQ_INSTS_VALU_INT32",
            "SQ_INSTS_VALU_INT64",
            "SQ_INSTS_VMEM_RD",
            "SQ_INSTS_VMEM_WR",
            "SQ_INST_CYCLES_VMEM_RD",
            "SQ_INST_CYCLES_VMEM_WR"
        };
    }

    // Full instruction mix
    static std::vector<std::string> full_instruction_mix() {
        return {
            "SQ_INSTS",
            "SQ_INSTS_VALU",
            "SQ_INSTS_SALU",
            "SQ_INSTS_VMEM",
            "SQ_INSTS_SMEM",
            "SQ_INSTS_LDS",
            "SQ_INSTS_BRANCH",
            "SQ_INSTS_MFMA",
            // FP64
            "SQ_INSTS_VALU_ADD_F64",
            "SQ_INSTS_VALU_MUL_F64",
            "SQ_INSTS_VALU_FMA_F64",
            "SQ_INSTS_VALU_TRANS_F64",
            "SQ_INSTS_VALU_MFMA_F64",
            // FP32
            "SQ_INSTS_VALU_ADD_F32",
            "SQ_INSTS_VALU_MUL_F32",
            "SQ_INSTS_VALU_FMA_F32",
            "SQ_INSTS_VALU_TRANS_F32",
            "SQ_INSTS_VALU_MFMA_F32",
            // FP16
            "SQ_INSTS_VALU_ADD_F16",
            "SQ_INSTS_VALU_MUL_F16",
            "SQ_INSTS_VALU_FMA_F16",
            "SQ_INSTS_VALU_MFMA_F16",
            // Other
            "SQ_INSTS_VALU_MFMA_BF16",
            "SQ_INSTS_VALU_MFMA_I8",
            "SQ_INSTS_VALU_INT32",
            "SQ_INSTS_VALU_INT64"
        };
    }
};

}  // namespace measure_flops
