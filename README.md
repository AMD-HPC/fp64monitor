# FP64 GPU Monitor

A C++ utility to monitor FP64 (double-precision) instruction usage on AMD GPUs using the rocprofiler-sdk library.

## Features

- Periodic hardware counter sampling for HIP applications
- FP64 instruction percentage analysis
- Multiple output formats: text, CSV, JSON
- Summary statistics (average, min, max FP64 percentages)
- Works via ROCP_TOOL_LIBRARIES for any HIP application
- MPI-aware output file naming (auto-detects rank)
- Build-time ROCm version recording
- Memory monitoring (GPU and CPU):
  - GPU memory usage via HIP API
  - Detailed VRAM/GTT breakdown via ROCm SMI (optional)
  - CPU resident and virtual memory via /proc/self/statm
  - High water mark tracking for peak memory usage

## Requirements

- ROCm 6.4+ or ROCm 7.x with rocprofiler-sdk
- CMake 3.21+
- C++17 compiler (amdclang++ from ROCm)

### ROCm Compatibility

This tool supports both ROCm 6.x and ROCm 7.x through a compatibility layer that handles API differences in rocprofiler-sdk:

| ROCm Version | rocprofiler-sdk | Status |
|--------------|-----------------|--------|
| 6.4.x        | 0.6.x           | Supported |
| 7.x          | 1.x             | Supported |

The tool automatically detects and uses the appropriate API at compile time.

### Optional: SYS_PERFMON

You may see this warning during execution:

```
Device could not be locked for profiling due to lack of permissions (capability SYS_PERFMON).
PMC Counters may be inaccurate and System Counter Collection will be degraded.
```

**Note**: Counter collection typically works despite this warning. If counters are all zeros, the issue is usually something else (see Troubleshooting)

## Building

```bash
# Source ROCm environment
source /path/to/rocm/sourceme.sh

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### Tool Library (Recommended)

Monitor any HIP application by preloading the tool library:

```bash
ROCP_TOOL_LIBRARIES=/path/to/libfp64_monitor_tool.so ./your_hip_app
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FP64_MONITOR_INTERVAL` | 100 | Sampling interval in milliseconds |
| `FP64_MONITOR_GPU` | 0 | GPU ID to monitor |
| `FP64_MONITOR_OUTPUT` | auto | Output file (see below) |
| `FP64_MONITOR_FORMAT` | csv | Output format: text, csv, json |
| `FP64_MONITOR_VERBOSE` | 0 | Verbose output (1=enabled) |

**Recommended interval**: Use intervals of 50-100ms for applications running less than a few seconds. For longer-running applications, 100-500ms is appropriate. The tool takes a final sample at exit to capture any remaining activity, but this may fail for very short applications.

### Output File Naming

By default, each process writes to a unique file:
- **With MPI**: `fp64_monitor_<rank>.csv` (e.g., `fp64_monitor_0.csv`)
- **Without MPI**: `fp64_monitor_<hostname>_<pid>.csv`

MPI rank is auto-detected from environment variables (SLURM_PROCID, OMPI_COMM_WORLD_RANK, PMI_RANK, etc.).

Custom patterns are supported:
- `%r` = MPI rank (or "norank" if not detected)
- `%p` = Process ID
- `%h` = Hostname

Use `FP64_MONITOR_OUTPUT=stdout` to write to console instead of file.

### Examples

```bash
# Basic monitoring (auto-generates unique CSV file per process)
ROCP_TOOL_LIBRARIES=./libfp64_monitor_tool.so ./my_app
# Creates: fp64_monitor_<hostname>_<pid>.csv

# MPI application (each rank gets unique file automatically)
mpirun -np 4 env ROCP_TOOL_LIBRARIES=./libfp64_monitor_tool.so ./my_mpi_app
# Creates: fp64_monitor_0.csv, fp64_monitor_1.csv, etc.

# Custom filename pattern for MPI
FP64_MONITOR_OUTPUT='output/rank_%r.csv' \
  mpirun -np 4 env ROCP_TOOL_LIBRARIES=./libfp64_monitor_tool.so ./my_mpi_app

# Text output to console
FP64_MONITOR_OUTPUT=stdout FP64_MONITOR_FORMAT=text \
  ROCP_TOOL_LIBRARIES=./libfp64_monitor_tool.so ./my_app

# JSON output with custom sampling interval
FP64_MONITOR_INTERVAL=50 FP64_MONITOR_FORMAT=json \
  ROCP_TOOL_LIBRARIES=./libfp64_monitor_tool.so ./my_app
```

### Standalone Commands

List available GPUs:
```bash
./measure_flops --list-gpus
```

List available counters:
```bash
./measure_flops --list-counters
```

## Output Formats

### Text Output

```
==============================================================================
FP64 GPU Monitor
==============================================================================
Sampling interval: 100 ms
Output format:     text
GPU IDs:           0
==============================================================================

Sample  Time(s)   GPU         FP64 VALU     Total VALU    FP64%     MFMA64%   
------------------------------------------------------------------------------
1       0.10      0           12345678      98765432      12.50     0.00      
2       0.20      0           12445678      99765432      12.47     0.00      
------------------------------------------------------------------------------

==============================================================================
Summary
==============================================================================
Total samples:      2
Duration:           0.10 seconds

FP64 VALU Percentage:
  Average:          12.48%
  Min:              12.47%
  Max:              12.50%
...
```

### CSV Output

```csv
sample_id,elapsed_seconds,gpu_id,total_instructions,valu_instructions,...,fp64_valu_pct,fp64_total_pct
1,0.100000,0,123456789,98765432,...,12.5000,1.1000
```

### JSON Output

```json
{
  "metadata": {
    "tool": "FP64 GPU Monitor",
    "version": "1.0.0",
    "start_time": "2026-04-01T10:30:00",
    "sampling_interval_ms": 100,
    "gpu_ids": [0]
  },
  "samples": [...],
  "summary": {
    "total_samples": 100,
    "duration_seconds": 10.0,
    "fp64_valu_pct": {"average": 12.5, "min": 10.0, "max": 15.0},
    ...
  }
}
```

## Counters Collected

The tool collects these FP64-related counters:

| Counter | Description |
|---------|-------------|
| `SQ_INSTS` | Total instructions |
| `SQ_INSTS_VALU` | Total VALU instructions |
| `SQ_INSTS_VALU_ADD_F64` | FP64 ADD/SUB |
| `SQ_INSTS_VALU_MUL_F64` | FP64 Multiply |
| `SQ_INSTS_VALU_FMA_F64` | FP64 FMA |
| `SQ_INSTS_VALU_TRANS_F64` | FP64 Transcendental |
| `SQ_INSTS_VALU_MFMA_F64` | FP64 Matrix FMA |
| `SQ_INSTS_MFMA` | Total MFMA instructions |

**Note**: Counter values are per-wavefront (64 threads), not per-thread.

## Test Programs

The `tests/` directory contains validation programs:

```bash
# Build everything including tests
cd build && make

# Run unit tests
./tests/test_instruction_analyzer

# Run FP64 test workload with monitoring
FP64_MONITOR_INTERVAL=50 ROCP_TOOL_LIBRARIES=./libfp64_monitor_tool.so ./tests/stream_triad_fp64

# Run full integration test suite
./tests/run_integration_tests.sh
```

### Available Tests

| Test | Purpose |
|------|---------|
| `test_instruction_analyzer` | Unit tests for FP64 percentage calculations |
| `stream_triad_fp64` | HIP workload with known FP64 instruction counts |
| `run_integration_tests.sh` | End-to-end integration tests |

## Interpreting Results

### Understanding FP64 Percentages

The tool reports several FP64 percentage metrics:

| Metric | Formula | What It Means |
|--------|---------|---------------|
| `fp64_valu_pct` | (FP64 VALU) / (Total VALU) | FP64 as % of all vector ALU work |
| `fp64_total_pct` | (FP64 All) / (Total Insts) | FP64 as % of all instructions |
| `fp64_compute_pct` | (FP64 All) / (VALU + SALU) | FP64 as % of compute-only |
| `mfma_fp64_pct` | (FP64 MFMA) / (Total MFMA) | FP64 as % of matrix operations |

### Why FP64% May Be Lower Than Expected

Even for a "pure FP64" workload, the FP64 percentage is typically **10-15%**, not 100%. This is because:

1. **Address calculations**: Array indexing generates integer ALU instructions
2. **Loop control**: Increment, compare, and branch instructions
3. **Boundary checks**: `if (idx < N)` generates comparison instructions
4. **Memory operations**: Loads/stores are not counted as FP64

For example, a STREAM Triad kernel (`a[i] = b[i] + c * d[i]`) on MI300A typically shows ~12.5% FP64 VALU percentage. This is the **true compute ratio**, not a measurement error.

### Cumulative vs Delta Values

Counter values are **cumulative** over the application's lifetime. Each sample shows total counts since the context started. To get per-interval rates:
- Use `compute_delta()` in the API
- Or post-process CSV: `delta = current_row - previous_row`

### Memory Monitoring

The tool monitors memory usage from multiple sources:

| Source | Columns | Description |
|--------|---------|-------------|
| HIP API | `gpu_mem_used_mb`, `gpu_mem_total_mb` | GPU memory from `hipMemGetInfo` |
| ROCm SMI | `vram_used_mb`, `vram_total_mb`, `gtt_used_mb`, `gtt_total_mb` | Detailed GPU memory breakdown (optional) |
| /proc/self/statm | `cpu_resident_mb`, `cpu_virtual_mb` | CPU memory usage |
| HWM | `*_hwm_mb` columns | High water marks (peak usage) |

**Notes:**
- ROCm SMI support requires `libdrm` development headers at build time
- Memory stats are collected even when counter sampling fails
- On APU systems (like MI300A), GPU and CPU share the same physical memory
- Virtual memory includes shared libraries and memory-mapped files

### Multi-XCC/XCD GPUs

On MI300 series GPUs with multiple XCCs (compute dies), counters are automatically summed across all XCCs. The reported values represent the total activity across the entire GPU.

## Troubleshooting

### Zero Counter Values

If all counter values are 0, check:

1. **GPU Architecture**: Ensure your code is compiled for the correct GPU architecture
   ```bash
   # Check your GPU
   rocm-smi --showproductname
   
   # Rebuild with correct architecture
   cmake .. -DCMAKE_HIP_ARCHITECTURES=gfx942  # For MI300
   ```

2. **Kernel execution**: Verify kernels are actually running
   ```bash
   # Test with rocprofv3
   rocprofv3 --hip-trace -- ./your_app
   ```

3. **ROCm environment**: Ensure ROCm is properly sourced

### Context Start Failures

If you see "Failed to start context", the tool automatically retries (up to 5 seconds) while waiting for HSA to load. This is normal for applications that take time to initialize.

### Counter Accuracy Warning

The `SYS_PERFMON` warning typically does not prevent counter collection. If you see this warning but get non-zero counter values, the counters are working correctly.

### Build Errors

If CMake cannot find rocprofiler-sdk:
```bash
# Ensure ROCm environment is sourced
source /opt/rocm/sourceme.sh  # or your ROCm path

# Specify path explicitly
cmake .. -DCMAKE_PREFIX_PATH=$ROCM_PATH
```

## Project Structure

```
measure_flops/
├── CMakeLists.txt                    # Main build configuration
├── README.md                         # This file
├── IMPLEMENTATION_PLAN.md            # Detailed design document
│
├── include/measure_flops/
│   ├── types.hpp                     # Core data structures
│   ├── counter_sampler.hpp           # Hardware counter sampling
│   ├── instruction_analyzer.hpp      # FP64 analysis
│   ├── output_formatter.hpp          # Output formatting
│   ├── memory_monitor.hpp            # GPU/CPU memory monitoring
│   └── rocprofiler_compat.hpp        # ROCm 6/7 compatibility layer
│
├── src/
│   ├── main.cpp                      # CLI executable
│   ├── counter_sampler.cpp
│   ├── instruction_analyzer.cpp
│   ├── output_formatter.cpp
│   └── memory_monitor.cpp            # Memory monitoring implementation
│
├── tools/
│   └── fp64_monitor_tool.cpp         # ROCP_TOOL_LIBRARIES tool library
│
└── tests/
    ├── CMakeLists.txt                # Test build configuration
    ├── README.md                     # Test documentation
    ├── stream_triad_fp64.hip         # FP64 validation workload
    ├── test_instruction_analyzer.cpp # Unit tests
    └── run_integration_tests.sh      # Integration test suite
```

## Version Information

The tool reports version information on startup:
```
[FP64 Monitor] v1.0.0 (built with ROCm 7.13.0a20260324-gfx94X)
[FP64 Monitor] Using rocprofiler-sdk v1.2.1 (1.2.1), priority=0
```

The ROCm version is captured at build time from the ROCm installation path.

## License

BSD 3-Clause License
