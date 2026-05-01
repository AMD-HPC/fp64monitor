# FP64 GPU Monitor - Implementation Plan

## Overview

A C++ utility to monitor HIP applications on AMD GPUs, measuring FP64 instruction usage and overall instruction mix using the rocprofiler-sdk library.

**Primary Goal**: Determine what percentage of an application's GPU compute is FP64-based through periodic hardware counter sampling.

---

## 1. Project Structure

```
measure_flops/
├── CMakeLists.txt                      # Main build configuration
├── README.md                           # Usage documentation
├── IMPLEMENTATION_PLAN.md              # This document
│
├── include/
│   └── measure_flops/
│       ├── types.hpp                   # Common types, enums, structs
│       ├── counter_sampler.hpp         # Hardware counter sampling interface
│       ├── instruction_analyzer.hpp    # Instruction mix analysis
│       └── output_formatter.hpp        # Output formatting interface
│
├── src/
│   ├── main.cpp                        # Entry point, CLI parsing
│   ├── counter_sampler.cpp             # rocprofiler-sdk integration
│   ├── instruction_analyzer.cpp        # FP64 percentage calculation
│   └── output_formatter.cpp            # CSV/JSON/console output
│
├── tools/
│   └── fp64_monitor_tool.cpp           # ROCP_TOOL_LIBRARIES tool version
│
└── tests/
    ├── CMakeLists.txt                  # Test build configuration
    ├── README.md                       # Test documentation
    ├── stream_triad_fp64.hip           # FP64 validation test (HIP)
    ├── validate_counters.sh            # Automated validation script
    ├── fp64_monitor_analyze_results.py # Counter analysis helper
    └── test_instruction_analyzer.cpp   # Unit tests
```

### Test Programs (Already Implemented)

The `tests/` directory contains validation programs with known FP64 instruction counts:

| Program | Purpose | Status |
|---------|---------|--------|
| `stream_triad_fp64.hip` | STREAM Triad with multiple FP64 kernel variants | ✓ Implemented |
| `validate_counters.sh` | Automated build and rocprofv3 validation | ✓ Implemented |
| `fp64_monitor_analyze_results.py` | Parse and analyze counter CSV output | ✓ Implemented |

**Validation Results (MI300A):**
```
Array size: 1,048,576 | Iterations: 10 | Kernel: FMA Triad
Expected SQ_INSTS_VALU_FMA_F64 per dispatch: 16,384
Actual SQ_INSTS_VALU_FMA_F64 per dispatch:   16,384 ✓
```

---

## 2. Dependencies and Build System

### 2.1 Required Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| rocprofiler-sdk | 1.0+ | Hardware counter access |
| CMake | 3.21+ | Build system |
| C++ Compiler | C++17 | amdclang++ from ROCm |
| Threads | pthread | Sampling thread |

### 2.2 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

# Find HIP compiler
if(NOT CMAKE_HIP_COMPILER)
    find_program(amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin)
    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(measure_flops VERSION 1.0.0 LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find rocprofiler-sdk
find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Main library
add_library(measure_flops_lib STATIC
    src/counter_sampler.cpp
    src/instruction_analyzer.cpp
    src/output_formatter.cpp
)
target_link_libraries(measure_flops_lib
    PUBLIC rocprofiler-sdk::rocprofiler-sdk
    PRIVATE Threads::Threads
)

# Standalone executable (for system-wide monitoring)
add_executable(measure_flops src/main.cpp)
target_link_libraries(measure_flops PRIVATE measure_flops_lib Threads::Threads)

# Tool library (for ROCP_TOOL_LIBRARIES injection)
add_library(fp64_monitor_tool SHARED tools/fp64_monitor_tool.cpp)
target_link_libraries(fp64_monitor_tool
    PRIVATE measure_flops_lib
            rocprofiler-sdk::rocprofiler-sdk
            Threads::Threads
)
set_target_properties(fp64_monitor_tool PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
```

### 2.3 Build Instructions

```bash
# Source ROCm environment
source /home/kevihuck/src/therock-tarball/sourceme-7.13.0a20260324-gfx94X.sh

# Build
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$ROCM_PATH -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## 3. Core Components

### 3.1 Types and Data Structures (`include/measure_flops/types.hpp`)

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <map>

namespace measure_flops {

// Counter categories for instruction mix analysis
enum class CounterCategory {
    FP64_VALU,      // F64 vector ALU (ADD, MUL, FMA, TRANS)
    FP64_MFMA,      // F64 matrix FMA
    FP32_VALU,      // F32 vector ALU
    FP32_MFMA,      // F32 matrix FMA
    FP16_VALU,      // F16 vector ALU
    FP16_MFMA,      // F16 matrix FMA
    BF16_MFMA,      // BF16 matrix FMA
    INT8_MFMA,      // INT8 matrix FMA
    INT32,          // 32-bit integer
    INT64,          // 64-bit integer
    MEMORY,         // Vector memory operations
    SCALAR,         // Scalar ALU + memory
    LDS,            // Local data share
    BRANCH,         // Branch instructions
    OTHER,          // Everything else
    TOTAL           // Total instructions
};

// Single counter value with metadata
struct CounterValue {
    std::string name;
    double value;
    uint64_t counter_id;
    std::map<std::string, size_t> dimensions;  // XCC, SE, etc.
};

// Sample containing all counter values at a point in time
struct Sample {
    std::chrono::steady_clock::time_point timestamp;
    uint64_t sample_id;
    int gpu_id;
    std::vector<CounterValue> counters;
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
    double fp32_total;
    double fp16_total;
    double bf16_mfma;
    double int8_mfma;
    double int32_total;
    double int64_total;
    
    // Memory and other
    double vmem_total;
    double smem_total;
    double lds_total;
    double salu_total;
    double branch_total;
    
    // Computed percentages
    double fp64_valu_pct;       // FP64 VALU as % of total VALU
    double fp64_total_pct;      // All FP64 as % of total instructions
    double fp64_compute_pct;    // FP64 as % of compute (excluding mem/branch)
    double mfma_fp64_pct;       // FP64 MFMA as % of all MFMA
};

// Configuration for the monitor
struct MonitorConfig {
    std::chrono::milliseconds sampling_interval{100};
    std::vector<int> gpu_ids;           // Empty = all GPUs
    std::string output_file;            // Empty = stdout
    std::string output_format{"text"};  // text, csv, json
    bool aggregate_dimensions{true};    // Sum across XCC/SE
    bool continuous{true};              // Run until interrupted
    int num_samples{0};                 // 0 = unlimited (if continuous)
    bool verbose{false};
};

}  // namespace measure_flops
```

### 3.2 Counter Sampler (`include/measure_flops/counter_sampler.hpp`)

```cpp
#pragma once

#include "measure_flops/types.hpp"
#include <rocprofiler-sdk/rocprofiler.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace measure_flops {

// GPU agent information
struct GpuAgent {
    rocprofiler_agent_id_t id;
    std::string name;
    std::string gfx_version;
    int logical_id;
};

class CounterSampler {
public:
    // Initialize sampler for a specific GPU agent
    explicit CounterSampler(const GpuAgent& agent);
    ~CounterSampler();
    
    // Non-copyable, movable
    CounterSampler(const CounterSampler&) = delete;
    CounterSampler& operator=(const CounterSampler&) = delete;
    CounterSampler(CounterSampler&&) noexcept;
    CounterSampler& operator=(CounterSampler&&) noexcept;
    
    // Get available GPU agents on the system
    static std::vector<GpuAgent> get_available_gpus();
    
    // Get list of all available counter names for this agent
    std::vector<std::string> get_available_counters() const;
    
    // Check if a specific counter is available
    bool is_counter_available(const std::string& name) const;
    
    // Configure which counters to collect
    // Returns false if any counter is unavailable
    bool configure_counters(const std::vector<std::string>& counter_names);
    
    // Take a single sample of all configured counters
    // Returns empty Sample on error
    Sample take_sample();
    
    // Get the GPU agent this sampler is configured for
    const GpuAgent& get_agent() const { return agent_; }
    
    // Get last error message
    std::string get_last_error() const { return last_error_; }

private:
    GpuAgent agent_;
    rocprofiler_context_id_t context_{};
    rocprofiler_buffer_id_t buffer_{};
    rocprofiler_counter_config_id_t profile_{.handle = 0};
    
    std::unordered_map<std::string, rocprofiler_counter_id_t> counter_name_to_id_;
    std::unordered_map<uint64_t, std::string> counter_id_to_name_;
    std::vector<std::string> configured_counters_;
    size_t expected_record_count_{0};
    uint64_t sample_counter_{0};
    std::string last_error_;
    bool context_initialized_{false};
    
    void initialize_context();
    void populate_counter_map();
    size_t get_counter_instance_count(rocprofiler_counter_id_t counter_id);
    void set_profile_callback(rocprofiler_context_id_t ctx,
                              rocprofiler_device_counting_agent_cb_t cb);
};

}  // namespace measure_flops
```

### 3.3 Instruction Analyzer (`include/measure_flops/instruction_analyzer.hpp`)

```cpp
#pragma once

#include "measure_flops/types.hpp"
#include <vector>

namespace measure_flops {

class InstructionAnalyzer {
public:
    // Analyze a sample to produce instruction mix breakdown
    static InstructionMix analyze(const Sample& sample, bool aggregate_dimensions = true);
    
    // Compute delta between two samples (for rate calculation)
    static InstructionMix compute_delta(const InstructionMix& current,
                                        const InstructionMix& previous);
    
    // Get the list of counters needed for full instruction mix analysis
    static std::vector<std::string> get_required_counters();
    
    // Get minimal counters for FP64 percentage only
    static std::vector<std::string> get_fp64_counters();

private:
    // Helper to sum counter values by name (across dimensions)
    static double sum_counter(const Sample& sample, const std::string& name);
    
    // Helper to get counter value without aggregation
    static std::vector<CounterValue> get_counter_instances(const Sample& sample,
                                                            const std::string& name);
};

}  // namespace measure_flops
```

### 3.4 Output Formatter (`include/measure_flops/output_formatter.hpp`)

```cpp
#pragma once

#include "measure_flops/types.hpp"
#include <ostream>
#include <memory>
#include <vector>

namespace measure_flops {

class OutputFormatter {
public:
    virtual ~OutputFormatter() = default;
    
    // Write header (for formats that need it)
    virtual void write_header(std::ostream& os) = 0;
    
    // Write a single instruction mix sample
    virtual void write_sample(std::ostream& os, const InstructionMix& mix) = 0;
    
    // Write footer/summary (for formats that need it)
    virtual void write_footer(std::ostream& os,
                              const std::vector<InstructionMix>& all_samples) = 0;
    
    // Factory method
    static std::unique_ptr<OutputFormatter> create(const std::string& format);
};

class TextFormatter : public OutputFormatter {
public:
    void write_header(std::ostream& os) override;
    void write_sample(std::ostream& os, const InstructionMix& mix) override;
    void write_footer(std::ostream& os,
                      const std::vector<InstructionMix>& all_samples) override;
};

class CsvFormatter : public OutputFormatter {
public:
    void write_header(std::ostream& os) override;
    void write_sample(std::ostream& os, const InstructionMix& mix) override;
    void write_footer(std::ostream& os,
                      const std::vector<InstructionMix>& all_samples) override;
};

class JsonFormatter : public OutputFormatter {
public:
    void write_header(std::ostream& os) override;
    void write_sample(std::ostream& os, const InstructionMix& mix) override;
    void write_footer(std::ostream& os,
                      const std::vector<InstructionMix>& all_samples) override;
private:
    bool first_sample_{true};
};

}  // namespace measure_flops
```

---

## 4. Counter Collection Strategy

### 4.1 Counter Groups

Due to hardware limitations, not all counters can be collected simultaneously. We organize counters into groups that can be collected together:

**Group 1: Core Instruction Counts**
```
SQ_INSTS                    # Total instructions
SQ_INSTS_VALU               # Total VALU instructions
SQ_INSTS_SALU               # Scalar ALU
SQ_INSTS_VMEM               # Vector memory
```

**Group 2: FP64 Instructions**
```
SQ_INSTS_VALU_ADD_F64       # F64 ADD/SUB
SQ_INSTS_VALU_MUL_F64       # F64 Multiply
SQ_INSTS_VALU_FMA_F64       # F64 FMA
SQ_INSTS_VALU_TRANS_F64     # F64 Transcendental
```

**Group 3: FP64 MFMA**
```
SQ_INSTS_VALU_MFMA_F64      # F64 MFMA instructions
SQ_INSTS_VALU_MFMA_MOPS_F64 # F64 MFMA operations (×512)
SQ_INSTS_MFMA               # Total MFMA instructions
```

**Group 4: Other Precision (Optional)**
```
SQ_INSTS_VALU_ADD_F32       # F32 ADD/SUB
SQ_INSTS_VALU_MUL_F32       # F32 Multiply
SQ_INSTS_VALU_FMA_F32       # F32 FMA
SQ_INSTS_VALU_TRANS_F32     # F32 Transcendental
SQ_INSTS_VALU_MFMA_F32      # F32 MFMA
```

**Group 5: FP16/BF16/INT (Optional)**
```
SQ_INSTS_VALU_ADD_F16       # F16 operations
SQ_INSTS_VALU_MUL_F16
SQ_INSTS_VALU_FMA_F16
SQ_INSTS_VALU_MFMA_F16
SQ_INSTS_VALU_MFMA_BF16
SQ_INSTS_VALU_MFMA_I8
SQ_INSTS_VALU_INT32
SQ_INSTS_VALU_INT64
```

**Group 6: Memory/Other (Optional)**
```
SQ_INSTS_SMEM               # Scalar memory
SQ_INSTS_LDS                # LDS operations
SQ_INSTS_BRANCH             # Branch instructions
SQ_INSTS_FLAT               # Flat memory operations
```

### 4.2 Sampling Strategy

**Option A: Single-Pass Minimal (Recommended for low overhead)**
- Collect Groups 1-3 in a single pass
- ~12 counters, should fit in one hardware pass
- Sufficient for FP64 percentage calculation

**Option B: Multi-Pass Complete**
- Collect all groups across multiple passes
- Higher overhead but complete instruction mix
- Round-robin through groups on each sample interval

### 4.3 Counter Value Interpretation

**IMPORTANT: SQ_INSTS counters count instructions per-wavefront (64 threads), NOT per-thread.**

Validated on MI300A with STREAM Triad FMA kernel:
```
Array size:     1,048,576 elements
Wavefronts:     1,048,576 / 64 = 16,384
Expected FMA:   16,384 per dispatch
Actual FMA:     16,384 per dispatch ✓
```

The FP64 percentage of VALU (~12.5% for Triad) is lower than expected because:
- Address calculations generate integer VALU instructions
- Boundary checks (if idx < n) generate comparison instructions
- This is the TRUE compute ratio, not a measurement error

Counters may also have dimensions (XCC, Shader Engine, etc.):
```cpp
// Example: SQ_INSTS_VALU_FMA_F64 on MI300X with 8 XCCs
// Returns 8 records, one per XCC
// To get total: sum all records with matching counter ID

double total_fp64_fma = 0;
for (const auto& record : records) {
    if (get_counter_name(record.id) == "SQ_INSTS_VALU_FMA_F64") {
        total_fp64_fma += record.counter_value;
    }
}
```

---

## 5. FP64 Percentage Calculation

### 5.1 Metrics

**FP64 VALU Percentage** (FP64 vector instructions as % of all VALU):
```
fp64_valu = SQ_INSTS_VALU_ADD_F64 + SQ_INSTS_VALU_MUL_F64 + 
            SQ_INSTS_VALU_FMA_F64 + SQ_INSTS_VALU_TRANS_F64

fp64_valu_pct = (fp64_valu / SQ_INSTS_VALU) × 100
```

**FP64 Total Percentage** (All FP64 as % of total instructions):
```
fp64_total = fp64_valu + SQ_INSTS_VALU_MFMA_F64

fp64_total_pct = (fp64_total / SQ_INSTS) × 100
```

**FP64 Compute Percentage** (FP64 as % of compute-only, excluding memory):
```
compute_insts = SQ_INSTS_VALU + SQ_INSTS_SALU

fp64_compute_pct = (fp64_total / compute_insts) × 100
```

**MFMA FP64 Percentage** (FP64 MFMA as % of all MFMA):
```
mfma_fp64_pct = (SQ_INSTS_VALU_MFMA_F64 / SQ_INSTS_MFMA) × 100
```

### 5.2 Delta Calculation

Since counters are cumulative, we compute deltas between samples:
```cpp
InstructionMix compute_delta(const InstructionMix& current, 
                             const InstructionMix& previous) {
    InstructionMix delta;
    delta.timestamp = current.timestamp;
    delta.fp64_add = current.fp64_add - previous.fp64_add;
    delta.fp64_mul = current.fp64_mul - previous.fp64_mul;
    // ... etc for all fields
    
    // Recompute percentages from deltas
    double fp64_valu = delta.fp64_add + delta.fp64_mul + 
                       delta.fp64_fma + delta.fp64_trans;
    delta.fp64_valu_pct = (delta.valu_instructions > 0) 
        ? (fp64_valu / delta.valu_instructions) * 100.0 
        : 0.0;
    // ... etc
    
    return delta;
}
```

---

## 6. CLI Interface

### 6.1 Usage

```
measure_flops [OPTIONS]

Options:
  -h, --help                Show this help message
  -v, --version             Show version information
  
  -i, --interval MS         Sampling interval in milliseconds (default: 100)
  -n, --samples N           Number of samples to collect (default: unlimited)
  -g, --gpu ID[,ID...]      GPU IDs to monitor (default: all)
  
  -o, --output FILE         Output file (default: stdout)
  -f, --format FORMAT       Output format: text, csv, json (default: text)
  
  --full                    Collect full instruction mix (higher overhead)
  --fp64-only               Collect only FP64-related counters (lower overhead)
  --list-counters           List all available counters and exit
  --list-gpus               List available GPUs and exit
  
  --verbose                 Verbose output including raw counter values
```

### 6.2 Example Usage

```bash
# Basic monitoring with default settings
./measure_flops

# Monitor for 10 seconds at 50ms intervals, output to CSV
./measure_flops -i 50 -n 200 -f csv -o results.csv

# Monitor specific GPUs with full instruction mix
./measure_flops -g 0,1 --full -f json -o results.json

# List available counters
./measure_flops --list-counters
```

### 6.3 Tool Mode (ROCP_TOOL_LIBRARIES)

```bash
# Monitor a specific HIP application
ROCP_TOOL_LIBRARIES=/path/to/libfp64_monitor_tool.so ./my_hip_app

# With custom settings via environment variables
FP64_MONITOR_INTERVAL=50 \
FP64_MONITOR_OUTPUT=results.csv \
FP64_MONITOR_FORMAT=csv \
ROCP_TOOL_LIBRARIES=/path/to/libfp64_monitor_tool.so ./my_hip_app
```

---

## 7. Output Formats

### 7.1 Text Format (Default)

```
================================================================================
FP64 GPU Monitor - Sample #1 @ 2026-04-01 10:30:00.123
GPU: 0 (gfx942)
================================================================================

Instruction Mix:
  Total Instructions:     1,234,567,890
  VALU Instructions:        987,654,321
  SALU Instructions:         98,765,432
  Memory Instructions:      148,148,137

FP64 Analysis:
  FP64 VALU (ADD/MUL/FMA/TRANS):  12,345,678  (1.25% of VALU)
  FP64 MFMA:                       1,234,567  (45.2% of all MFMA)
  
  FP64 Total:                     13,580,245
  FP64 % of Total Instructions:        1.10%
  FP64 % of Compute:                   1.35%

Breakdown by Precision:
  FP64:   1.25%  ████
  FP32:  45.67%  ██████████████████████████████████████████████
  FP16:  23.45%  ████████████████████████
  INT:   15.34%  ███████████████
  Other: 14.29%  ██████████████
--------------------------------------------------------------------------------
```

### 7.2 CSV Format

```csv
timestamp,sample_id,gpu_id,total_insts,valu_insts,fp64_add,fp64_mul,fp64_fma,fp64_trans,fp64_mfma,fp64_valu_pct,fp64_total_pct,fp64_compute_pct
2026-04-01T10:30:00.123,1,0,1234567890,987654321,1234567,2345678,3456789,4567890,1234567,1.25,1.10,1.35
2026-04-01T10:30:00.223,2,0,1334567890,1087654321,1334567,2445678,3556789,4667890,1334567,1.26,1.11,1.36
```

### 7.3 JSON Format

```json
{
  "metadata": {
    "tool_version": "1.0.0",
    "rocm_version": "7.13.0",
    "start_time": "2026-04-01T10:30:00.000",
    "sampling_interval_ms": 100,
    "gpus": [
      {"id": 0, "name": "gfx942", "agent_id": 12345}
    ]
  },
  "samples": [
    {
      "timestamp": "2026-04-01T10:30:00.123",
      "sample_id": 1,
      "gpu_id": 0,
      "counters": {
        "total_instructions": 1234567890,
        "valu_instructions": 987654321,
        "fp64_add": 1234567,
        "fp64_mul": 2345678,
        "fp64_fma": 3456789,
        "fp64_trans": 4567890,
        "fp64_mfma": 1234567
      },
      "analysis": {
        "fp64_valu_pct": 1.25,
        "fp64_total_pct": 1.10,
        "fp64_compute_pct": 1.35,
        "mfma_fp64_pct": 45.2
      }
    }
  ],
  "summary": {
    "total_samples": 100,
    "duration_seconds": 10.0,
    "average_fp64_pct": 1.23,
    "peak_fp64_pct": 2.45,
    "min_fp64_pct": 0.12
  }
}
```

---

## 8. Implementation Steps

### Phase 1: Foundation (Core Infrastructure) ✓ COMPLETE

1. **Set up project structure and build system** ✓
   - Create directory structure
   - Write CMakeLists.txt
   - Verify build with ROCm environment

2. **Implement `types.hpp`** ✓
   - Define all data structures
   - Define counter category enums

3. **Implement basic `CounterSampler`** ✓
   - GPU agent discovery
   - Counter enumeration
   - Context initialization

### Phase 2: Core Functionality ✓ COMPLETE

4. **Complete `CounterSampler` implementation** ✓
   - Counter configuration
   - Sample collection via `rocprofiler_sample_device_counting_service()`
   - Error handling

5. **Implement `InstructionAnalyzer`** ✓
   - Counter aggregation across dimensions
   - FP64 percentage calculations
   - Delta computation

6. **Implement `OutputFormatter`** ✓
   - Text formatter
   - CSV formatter
   - JSON formatter

### Phase 3: CLI and Integration ✓ COMPLETE

7. **Implement `main.cpp`** ✓
   - CLI argument parsing
   - Main sampling loop
   - Signal handling (SIGINT)
   - Output file handling

8. **Implement tool library (`fp64_monitor_tool.cpp`)** ✓
   - rocprofiler tool registration
   - Background sampling thread
   - Environment variable configuration
   - MPI-aware output file naming
   - Build-time ROCm version recording

### Phase 4: Testing and Documentation ✓ COMPLETE

9. **Testing** ✓
   - Unit tests for InstructionAnalyzer (`test_instruction_analyzer.cpp`)
   - Integration tests with sample HIP programs (`run_integration_tests.sh`)
   - Validation against rocprof output (`validate_counters.sh`)

10. **Documentation** ✓
    - README with usage examples
    - Build instructions
    - Interpretation guide for results

---

## 9. Error Handling

### 9.1 Common Errors

| Error | Cause | Handling |
|-------|-------|----------|
| `ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED` | HSA runtime not initialized | Retry after delay |
| `ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND` | Invalid counter name | Report and skip |
| `ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED` | Context already started | Stop context first |
| `ROCPROFILER_STATUS_ERROR_AGENT_MISMATCH` | Profile/context agent mismatch | Recreate profile |

### 9.2 Error Reporting

```cpp
#define ROCPROFILER_CHECK(result, msg) \
    do { \
        rocprofiler_status_t status = (result); \
        if (status != ROCPROFILER_STATUS_SUCCESS) { \
            throw std::runtime_error( \
                std::string(msg) + ": " + \
                rocprofiler_get_status_string(status)); \
        } \
    } while(0)
```

---

## 10. Performance Considerations

### 10.1 Overhead Estimation

| Sampling Interval | Expected Overhead |
|-------------------|-------------------|
| 1000ms            | < 0.1%            |
| 100ms             | < 1%              |
| 50ms              | 1-2%              |
| 10ms              | 5-10%             |

### 10.2 Minimizing Overhead

1. **Counter selection**: Collect only necessary counters
2. **Aggregation**: Aggregate dimensions in post-processing
3. **Buffering**: Buffer output writes
4. **Async sampling**: Use `ROCPROFILER_COUNTER_FLAG_ASYNC` when possible

---

## 11. Testing Plan

### 11.1 Unit Tests

- `test_instruction_analyzer.cpp`: Test FP64 percentage calculations
- `test_output_formatter.cpp`: Test output format correctness

### 11.2 Integration Tests

```bash
# Test with synthetic workload
./measure_flops -n 10 -i 100 -f json -o /tmp/test.json &
./synthetic_fp64_workload
wait
# Validate output

# Test with rocblas DGEMM (known FP64 workload)
./measure_flops -n 100 -i 50 &
rocblas-bench -f gemm -r d -m 4096 -n 4096 -k 4096
# Expect high FP64 percentage
```

### 11.3 Validation

Compare results against `rocprofv3`:
```bash
# Our tool
./measure_flops --fp64-only -n 10 -f csv -o our_results.csv &
./test_app &
wait

# rocprofv3
rocprofv3 --pmc "SQ_INSTS SQ_INSTS_VALU SQ_INSTS_VALU_FMA_F64" -o rocprof_results -- ./test_app

# Compare values
```

---

## 12. Future Enhancements

1. **Per-kernel breakdown**: Track counters per kernel dispatch
2. **Real-time visualization**: Terminal-based live graphs
3. **Process filtering**: Monitor specific PIDs only
4. **Power correlation**: Correlate FP64 usage with power consumption
5. **MPI support**: Aggregate across MPI ranks
6. **Prometheus export**: Export metrics for monitoring systems

---

## 13. References

- ROCprofiler-SDK Documentation: https://rocm.docs.amd.com/projects/rocprofiler-sdk/
- MI300 Performance Counters: https://rocm.docs.amd.com/en/docs-6.4.1/conceptual/gpu-arch/mi300-mi200-performance-counters.html
- Sample Code: `$ROCM_PATH/share/rocprofiler-sdk/samples/counter_collection/`
