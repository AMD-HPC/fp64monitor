# FP64 GPU Monitor

A C++ utility to monitor GPU instruction usage on AMD GPUs using the rocprofiler-sdk library. While named for FP64 monitoring, it supports multiple counter sets for analyzing different aspects of GPU workloads.

## Features

- Periodic hardware counter sampling for HIP applications
- Multiple counter sets: FP64, FP32, FP16, INT, SALU, and high-level instruction mix
- Multiple output formats: text, CSV, JSON
- Works via `ROCP_TOOL_LIBRARIES` for any HIP application
- MPI-aware output file naming (auto-detects rank)
- Build-time ROCm version recording
- Memory monitoring (GPU and CPU):
  - Detailed VRAM/GTT breakdown via AMD SMI
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

### Optional: `SYS_PERFMON`

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

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_AMD_SMI` | ON | Enable AMD SMI memory monitoring (requires `libamd_smi`) |
| `BUILD_TESTS` | ON | Build test programs |

Example without AMD SMI support:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_AMD_SMI=OFF
```

**Note**: If `--use-amdsmi` is requested at runtime but the tool was built without AMD SMI support, an error message will be displayed and the tool will continue without memory monitoring features.

## Usage

### Wrapper Script (Recommended)

The easiest way to use `fp64_monitor` is with the wrapper script:

```bash
# Basic usage
fp64_monitor.sh -- ./your_hip_app [args...]

# With options
fp64_monitor.sh --interval 500 --verbose -- ./your_app arg1 arg2

# MPI applications with memory monitoring
mpirun -np 4 fp64_monitor.sh --use-amdsmi --tmpdir /dev/shm -- ./my_mpi_app

# High-level instruction mix analysis
fp64_monitor.sh --counters highlevel --use-amdsmi -- ./my_app
```

#### Wrapper Script Options

| Option | Description |
|--------|-------------|
| `--interval MS` | Sampling interval in ms (default: 1000) |
| `--output PATTERN` | Output file pattern (default: auto) |
| `--format FORMAT` | Output format: csv, json, text (default: csv) |
| `--dir DIR` | Output directory (default: `fp64_monitor_output`) |
| `--gpu ID` | GPU ID to monitor (default: 0) |
| `--tmpdir DIR` | Fast local storage for temp output (e.g., /dev/shm) |
| `--use-amdsmi` | Enable AMD SMI memory monitoring (VRAM/GTT) |
| `--preload` | Wrap main function for clean shutdown (auto-enabled with --use-amdsmi) |
| `--counters TYPE` | Counter set: fp64, fp32, fp16, int, salu, highlevel, highlevel2 |
| `--no-analyze` | Disable auto-analysis after run |
| `--verbose` | Enable verbose output and rocprofiler-sdk logging |
| `--help` | Show brief help message |
| `--fullhelp` | Show detailed help with counter descriptions and examples |

### Tool Library (Direct)

Monitor any HIP application by setting the tool library directly:

```bash
ROCP_TOOL_LIBRARIES=/path/to/libfp64_monitor_tool.so ./your_hip_app
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FP64_MONITOR_INTERVAL` | 1000 | Sampling interval in milliseconds |
| `FP64_MONITOR_GPU` | 0 | GPU ID to monitor |
| `FP64_MONITOR_OUTPUT` | auto | Output file pattern (see below) |
| `FP64_MONITOR_FORMAT` | csv | Output format: text, csv, json |
| `FP64_MONITOR_DIR` | `fp64_monitor_output` | Output directory |
| `FP64_MONITOR_TMPDIR` | (none) | Fast local storage for temp files (e.g., /dev/shm) |
| `FP64_MONITOR_COUNTERS` | fp64 | Counter set: fp64, fp32, fp16, int, salu, highlevel, highlevel2 |
| `FP64_MONITOR_USE_AMDSMI` | 0 | Enable AMD SMI memory monitoring (1=enabled) |
| `FP64_MONITOR_PRELOAD` | 0 | Enable `LD_PRELOAD` wrapper for clean shutdown (1=enabled) |
| `FP64_MONITOR_ANALYZE` | 1 | Auto-analyze results after run (0=disabled) |
| `FP64_MONITOR_VERBOSE` | 0 | Verbose output (1=enabled) |

**Recommended interval**: Use intervals of 50-100ms for applications running less than a few seconds. For longer-running applications, 500-1000ms is appropriate. The tool takes a final sample at exit to capture any remaining activity.

### Output File Naming

By default, each process writes to a unique file:
- **With MPI**: `fp64_monitor_<rank>.csv` (e.g., `fp64_monitor_0.csv`)
- **Without MPI**: `fp64_monitor_<hostname>_<pid>.csv`

MPI rank is auto-detected from environment variables (`SLURM_PROCID`, `OMPI_COMM_WORLD_RANK`, `PMI_RANK`, etc.).

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

## Counter Sets

The tool supports multiple counter sets, selected via `--counters TYPE` or `FP64_MONITOR_COUNTERS`. Hardware limits restrict simultaneous counter collection to ~8 SQ-block counters, which is why different counter sets exist.

### FP64 Counters (default)

| Counter | Description |
|---------|-------------|
| `SQ_INSTS` | Total instructions |
| `SQ_INSTS_VALU` | Total VALU instructions |
| `SQ_INSTS_VALU_ADD_F64` | FP64 ADD/SUB |
| `SQ_INSTS_VALU_MUL_F64` | FP64 Multiply |
| `SQ_INSTS_VALU_FMA_F64` | FP64 FMA |
| `SQ_INSTS_VALU_TRANS_F64` | FP64 Transcendental |
| `SQ_INSTS_MFMA` | Total MFMA instructions |
| `SQ_INSTS_MFMA_MOPS_F64` | FP64 Matrix FMA operations |

### FP32 Counters (`--counters fp32`)

| Counter | Description |
|---------|-------------|
| `SQ_INSTS` | Total instructions |
| `SQ_INSTS_VALU` | Total VALU instructions |
| `SQ_INSTS_MFMA` | Total MFMA instructions |
| `SQ_INSTS_VALU_ADD_F32` | FP32 ADD/SUB |
| `SQ_INSTS_VALU_MUL_F32` | FP32 Multiply |
| `SQ_INSTS_VALU_FMA_F32` | FP32 FMA |
| `SQ_INSTS_VALU_TRANS_F32` | FP32 Transcendental |
| `SQ_INSTS_MFMA_MOPS_F32` | FP32 Matrix FMA operations |

### FP16 Counters (`--counters fp16`)

| Counter | Description |
|---------|-------------|
| `SQ_INSTS` | Total instructions |
| `SQ_INSTS_VALU` | Total VALU instructions |
| `SQ_INSTS_MFMA` | Total MFMA instructions |
| `SQ_INSTS_VALU_ADD_F16` | FP16 ADD/SUB |
| `SQ_INSTS_VALU_MUL_F16` | FP16 Multiply |
| `SQ_INSTS_VALU_FMA_F16` | FP16 FMA |
| `SQ_INSTS_MFMA_MOPS_F16` | FP16 Matrix FMA operations |
| `SQ_INSTS_MFMA_MOPS_BF16` | BF16 Matrix FMA operations |

### Integer Counters (`--counters int`)

| Counter | Description |
|---------|-------------|
| `SQ_INSTS` | Total instructions |
| `SQ_INSTS_VALU` | Total VALU instructions |
| `SQ_INSTS_VALU_INT32` | 32-bit integer VALU |
| `SQ_INSTS_VALU_INT64` | 64-bit integer VALU |
| `SQ_INSTS_MFMA` | Total MFMA instructions |
| `SQ_INSTS_MFMA_MOPS_I8` | INT8 Matrix FMA operations |

### SALU Counters (`--counters salu`)

| Counter | Description |
|---------|-------------|
| `SQ_INSTS` | Total instructions |
| `SQ_INSTS_SALU` | Scalar ALU instructions |
| `SQ_INSTS_SMEM` | Scalar memory instructions |
| `SQ_INSTS_FLAT` | Flat memory instructions |
| `SQ_INSTS_LDS` | Local data share instructions |
| `SQ_INSTS_BRANCH` | Branch instructions |

### High-Level Instruction Mix (`--counters highlevel`)

Provides a broad overview of instruction categories:

| Counter | Description |
|---------|-------------|
| `SQ_INSTS` | Total instructions |
| `SQ_INSTS_VALU` | Vector ALU instructions |
| `SQ_INSTS_SALU` | Scalar ALU instructions |
| `SQ_INSTS_VMEM` | Vector memory instructions |
| `SQ_INSTS_SMEM` | Scalar memory instructions |
| `SQ_INSTS_LDS` | Local data share instructions |
| `SQ_INSTS_BRANCH` | Branch instructions |
| `SQ_INSTS_MFMA` | Matrix FMA instructions |

### Other Instruction Types (`--counters highlevel2`)

| Counter | Description |
|---------|-------------|
| `SQ_INSTS` | Total instructions |
| `SQ_INSTS_FLAT` | Flat memory instructions |
| `SQ_INSTS_GDS` | Global data share instructions |
| `SQ_INSTS_EXP_GDS` | Export GDS instructions |
| `SQ_INSTS_SENDMSG` | Send message instructions |
| `SQ_INSTS_VSKIPPED` | Skipped VALU instructions |

**Note**: Counter values are per-wavefront (64 threads), not per-thread.

## Analysis Tools

### `analyze_results.py`

Aggregates CSV output files and generates a summary report. Automatically run by the wrapper script after the application exits (disable with `--no-analyze`).

```bash
# Manual usage
python3 analyze_results.py fp64_monitor_output/fp64_monitor_*.csv
```

Features:
- Auto-detects counter set from CSV data
- Aggregates data across multiple MPI ranks
- Reports instruction counts specific to the counter set used
- Memory high-water marks (VRAM, GTT, CPU) when available

### `plot_timeseries.py`

Generates time-series visualizations of the instruction mix over time.

```bash
# Generate PDF charts (auto-detects counter set)
python3 plot_timeseries.py fp64_monitor_output/fp64_monitor_*.csv

# Generate interactive HTML charts (requires plotly)
python3 plot_timeseries.py --html fp64_monitor_output/fp64_monitor_*.csv

# Explicitly specify counter set
python3 plot_timeseries.py --counters highlevel fp64_monitor_output/fp64_monitor_*.csv
```

Options:
| Option | Description |
|--------|-------------|
| `--output PREFIX` | Output file prefix (default: derived from input) |
| `--html` | Generate interactive HTML instead of PDF |
| `--counters SET` | Counter set: fp64, fp32, fp16, int, salu, highlevel, highlevel2, or auto |

Features:
- Supports all counter sets (auto-detection by default)
- Raw instruction counts with breakdown (stacked area chart)
- Percentage relative to total instructions
- Average values across MPI ranks with variability bands
- Interactive HTML output with zoom/hover (requires plotly)

### Python Dependencies

Install required packages (for plot\_timeseries.py):

```bash
pip install -r requirements.txt
```

Required: `matplotlib>=3.5.0` (for PDF output)
Optional: `plotly>=5.0.0` (for HTML output)

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

### Cumulative Counter Values

Counter values in the CSV output are **cumulative** over the application's lifetime. Each sample shows total counts since profiling started. To get per-interval rates:
- Post-process CSV: `delta = current_row - previous_row`
- The analysis scripts (`analyze_results.py`, `plot_timeseries.py`) handle this automatically

### Memory Monitoring

The tool monitors memory usage from multiple sources when `--use-amdsmi` is enabled:

| Source | Columns | Description |
|--------|---------|-------------|
| AMD SMI | `vram_used_mb`, `vram_total_mb` | GPU VRAM usage |
| AMD SMI | `gtt_used_mb`, `gtt_total_mb` | GPU GTT (system memory accessible by GPU) |
| /proc/self/statm | `cpu_resident_mb`, `cpu_virtual_mb` | CPU memory usage |
| High Water Marks | `*_hwm_mb` columns | Peak usage during application run |

**Notes:**
- AMD SMI support is enabled by default at build time (requires `libamd_smi`)
- Memory stats are collected even when counter sampling fails
- On APU systems (like MI300A), GPU and CPU share the same physical memory
- Use `--use-amdsmi` to enable GPU memory monitoring at runtime

### Multi-XCC/XCD GPUs

On MI300 series GPUs with multiple XCCs (compute dies), counters are automatically summed across all XCCs. The reported values represent the total activity across the entire GPU.

## Troubleshooting

### Zero Counter Values

If all counter values are 0, check:

1. **GPU Architecture**: Ensure your code is compiled for the correct GPU architecture
   ```bash
   # Check your GPU
   amd-smi static --asic   # or: rocminfo | grep gfx
   
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
├── requirements.txt                  # Python dependencies for analysis scripts
│
├── include/measure_flops/
│   ├── types.hpp                     # Core data structures and counter sets
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
│   ├── fp64_monitor.sh               # Wrapper script (recommended entry point)
│   ├── fp64_monitor_tool.cpp         # ROCP_TOOL_LIBRARIES tool library
│   └── preload.cpp                   # LD_PRELOAD helper for clean shutdown
│
├── scripts/
│   └── build.sh                      # Build helper script
│
└── tests/
    ├── CMakeLists.txt                # Test build configuration
    ├── README.md                     # Test documentation
    ├── analyze_results.py            # Post-processing analysis script
    ├── plot_timeseries.py            # Time-series visualization (PDF/HTML)
    ├── stream_triad_fp64.hip         # FP64 validation workload
    ├── test_instruction_analyzer.cpp # Unit tests
    ├── test_mfma_direct.cpp          # MFMA instruction test
    ├── test_rocblas_gemm.cpp         # rocBLAS GEMM test
    ├── validate_counters.sh          # Counter validation script
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
