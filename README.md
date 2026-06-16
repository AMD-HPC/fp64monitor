# FP64 GPU Monitor

C++ tooling (rocprofiler-sdk) to sample AMD GPU hardware counters while any HIP application runs. The default focus is FP64 instruction mix, but several counter sets (FP32, FP16, INT, SALU, high-level mix, and more) are supported. VRAM/GTT-style GPU memory stats (via AMD SMI) are available only when the project is configured with **`ENABLE_AMD_SMI=ON`**; CPU `/proc` memory fields still work without it.

## Requirements

- ROCm 6.4+ or 7.x with **rocprofiler-sdk** (0.6.x on ROCm 6, 1.x on ROCm 7; the code selects APIs at compile time)
- CMake 3.21+, C++17, HIP (**amdclang++** from ROCm is typical)
- Optional: **libamd_smi** for GPU memory stats (`ENABLE_AMD_SMI`, off by default)

You may see a `SYS_PERFMON` capability warning; counters usually still work. If values are all zero, see [Troubleshooting](#troubleshooting).

## Build

```bash
source /path/to/rocm/sourceme.sh   # or set ROCM_PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

CMake options: **`ENABLE_AMD_SMI`** (default **OFF**; set **`-DENABLE_AMD_SMI=ON`** to link **libamd_smi** and enable GPU memory columns for `--use-amdsmi` / `FP64_MONITOR_USE_AMDSMI`), **`BUILD_TESTS`** (default **OFF**; set **`-DBUILD_TESTS=ON`** for HIP test binaries and `ctest`). GPU arch for HIP tests defaults in `tests/CMakeLists.txt`; override with `-DCMAKE_HIP_ARCHITECTURES=gfx942` (or your ASIC) if needed.

Install (recommended for the wrapper script, which expects `bin/` + `lib/`):

```bash
cmake --install build --prefix /your/install/prefix
```

### Profiling library name (ROCm major suffix)

The injected shared library is built as CMake target `fp64_monitor_tool_<major>` and installed as:

`lib/libfp64_monitor_tool_<major>.so`

`<major>` is the **ROCm major** only (e.g. `6`, `7`), derived at **configure** time from the `ROCM_PATH` basename (e.g. `rocm-7.13.0a...` → `7`) or from HIP’s reported version if the path has no digits. The built tool is installed as **`libfp64_monitor_tool_<major>.so`** next to `prefix/lib/`.

- **`fp64_monitor.sh --rocm <major>`** is **required** and must match that installed library (you may pass `7` or `7.13`; only the major is used).
- For **`ROCP_TOOL_LIBRARIES`** by hand, use the exact filename under your build or install `lib/` (e.g. `ls build/libfp64_monitor_tool_*.so`).

The **`measure_flops`** binary only lists GPUs and counters; live monitoring uses the versioned tool `.so` above.

## Usage

Configure and build on a machine that matches where you will run (GPU present, ROCm loaded, **`CMAKE_HIP_ARCHITECTURES`** / **`GPU_TARGETS`** aligned with the ASIC). On shared clusters, batch and interactive jobs usually run on compute nodes; use your center’s documented way to obtain a GPU allocation. After **`cmake --install`**, put **`prefix/bin`** on **`PATH`** or invoke **`fp64_monitor.sh`** by full path so it can resolve **`../lib/libfp64_monitor_tool_<major>.so`** next to your install layout.

**Wrapper (recommended after install):**

```bash
fp64_monitor.sh --rocm 7 -- ./your_hip_app [args...]
fp64_monitor.sh --rocm 7 --interval 500 --counters highlevel --use-amdsmi -- ./your_app
mpirun -np 4 fp64_monitor.sh --rocm 7 --use-amdsmi --tmpdir /dev/shm -- ./mpi_app
```

Use the same wrapper in whatever parallel or batch launcher your system provides (for example Slurm **`srun`**, Flux **`flux run`**, or the **`mpiexec`** line from your job script): each MPI rank should execute **`fp64_monitor.sh`** with the same **`--rocm`** and options, then **`--`**, then your application and its arguments.

`fp64_monitor.sh --help` / `--fullhelp` document options (**`--rocm`** is required, interval, format, output pattern, **`--gpus`**, counter set, tmpdir, AMD SMI, preload, analysis). **`--use-amdsmi`** only affects GPU VRAM/GTT reporting when the tool was built with **`ENABLE_AMD_SMI=ON`**; otherwise rank 0 prints a short message and monitoring continues without AMD SMI.

**GPU selection (wrapper):** **`--gpus IDS`** (comma-separated logical IDs, or **`all`** for every GPU visible after ROCR) **>** **`FP64_MONITOR_GPUS`** **>** **`ROCR_VISIBLE_DEVICES`** **>** default **`0`**. Use **`--gpus`** (e.g. **`--gpus 2`** or **`--gpus 0,1`**) to override ROCR when you only want to monitor a subset. **Multiple GPUs:** one output file per GPU (default names include **`_gpu<N>_`**). Custom **`FP64_MONITOR_OUTPUT`** with several GPUs must include **`%g`** (or the wrapper inserts **`_gpu%g`** before the extension) so paths stay unique.

**`fp64_monitor.sh` behavior (install layout):** Resolves `../lib/libfp64_monitor_tool_<major>.so` from the script’s directory (using **`--rocm`**) and sets **`FP64_MONITOR_GPUS`** per the precedence above when **`ROCR_VISIBLE_DEVICES`** lists devices (e.g. `0,1,2,3`). Unless **`--verbose`**, sets **`GLOG_minloglevel=3`** to quiet rocprofiler-sdk logging. After the app exits, **`LD_PRELOAD`** is restored; if auto-analysis is on and this process is MPI rank **0** (or non-MPI), it waits **5 seconds** then runs **`fp64_monitor_analyze_results.py`** (searching `bin/`, repo `tests/`, etc.) on `fp64_monitor_*.csv` under **`FP64_MONITOR_DIR`**. Under **Flux**, the default output directory can get a job-id suffix when Flux is detected at startup.

**Direct injection:**

```bash
export ROCP_TOOL_LIBRARIES=/path/to/lib/libfp64_monitor_tool_<major>.so
./your_hip_app
```

**Environment** (when not using the wrapper): `FP64_MONITOR_INTERVAL`, **`FP64_MONITOR_GPUS`** (comma-separated logical IDs or **`all`**), `FP64_MONITOR_OUTPUT`, `FP64_MONITOR_FORMAT` (`csv`/`json`/`text`), `FP64_MONITOR_DIR`, `FP64_MONITOR_TMPDIR`, `FP64_MONITOR_COUNTERS`, `FP64_MONITOR_USE_AMDSMI`, `FP64_MONITOR_PRELOAD`, `FP64_MONITOR_ANALYZE`, `FP64_MONITOR_VERBOSE`. Output naming supports MPI rank detection and patterns with `%r`, `%p`, `%h`; use `FP64_MONITOR_OUTPUT=stdout` for the console.

Shorter runs: try **50–100 ms** intervals; longer jobs often use **500–1000 ms**. Samples are **cumulative** totals; use the Python tools or deltas between rows for per-interval rates.

**Counter sets:** `fp64` (default), `fp32`, `fp16`, `int`, `salu`, `highlevel`, `highlevel2`. Names match `FP64_MONITOR_COUNTERS`. Per-counter hardware names and limits are in `--fullhelp` and in `IMPLEMENTATION_PLAN.md`.

**Utility CLI:**

```bash
measure_flops --list-gpus
measure_flops --list-counters -g 0
```

## Analysis and plotting

After a run, the wrapper runs `fp64_monitor_analyze_results.py` on the output directory unless `--no-analyze` is set. Scripts live in `tests/` in the source tree and are installed to `bin/`:

```bash
python3 tests/fp64_monitor_analyze_results.py fp64_monitor_output/fp64_monitor_*.csv
python3 tests/fp64_monitor_plot_timeseries.py fp64_monitor_output/fp64_monitor_*.csv   # PDF
python3 tests/fp64_monitor_plot_timeseries.py --html fp64_monitor_output/fp64_monitor_*.csv
```

`pip install -r requirements.txt` — **matplotlib** for PDF; **plotly** optional for HTML.

## Tests

With `BUILD_TESTS` on, from the build directory:

```bash
./tests/test_instruction_analyzer
./tests/run_integration_tests.sh
# Example monitored workload:
FP64_MONITOR_INTERVAL=50 ROCP_TOOL_LIBRARIES="$PWD/libfp64_monitor_tool_<major>.so" ./tests/stream_triad_fp64
```

Other HIP targets include `stream_triad_fp32`, `test_mfma_direct`, `test_spmv`, and optionally `test_mfma_rocwmma` when rocWMMA is found. See `tests/README.md`.

## Example output

```bash
./build/bin/fp64_monitor.sh --rocm 7 --interval 100 -- ./build/tests/stream_triad_fp64
```

```text
########################################################
#          STREAM Triad FP64 Test Program              #
########################################################

This program generates known FP64 instruction counts
for validating hardware counter collection.

Device: AMD Instinct MI300A
  Compute Units: 228
  Clock Rate:    2100 MHz
  Memory:        128 GB

Configuration:
  Array size:   1048576 elements (8 MB per array)
  Iterations:   100
  Grid:         4096 blocks x 256 threads
  Scalar value: 3

======================================================================
Kernel: STREAM Triad (standard)
----------------------------------------------------------------------
Array size:              1048576 elements
Iterations:                  100 (+ 5 warmup)
Wavefronts/iter:           16384 (elements/64)
----------------------------------------------------------------------
Expected counter values (per-wavefront, cumulative):
  SQ_INSTS_VALU_ADD_F64:               0
  SQ_INSTS_VALU_MUL_F64:               0
  SQ_INSTS_VALU_FMA_F64:         1720320
  Total FP64 VALU:               1720320
======================================================================

Running 100 iterations...

Performance:
  Time:      0.54 ms
  Bandwidth: 4641.40 GB/s
  GFLOP/s:   386.78

======================================================================
Kernel: STREAM Triad (explicit FMA)
----------------------------------------------------------------------
Array size:              1048576 elements
Iterations:                  100 (+ 5 warmup)
Wavefronts/iter:           16384 (elements/64)
----------------------------------------------------------------------
Expected counter values (per-wavefront, cumulative):
  SQ_INSTS_VALU_ADD_F64:               0
  SQ_INSTS_VALU_MUL_F64:               0
  SQ_INSTS_VALU_FMA_F64:         1720320
  Total FP64 VALU:               1720320
======================================================================

Running 100 iterations...

Performance:
  Time:      0.59 ms
  Bandwidth: 4287.54 GB/s
  GFLOP/s:   357.29

======================================================================
Kernel: STREAM Triad (separate MUL+ADD)
----------------------------------------------------------------------
Array size:              1048576 elements
Iterations:                  100 (+ 5 warmup)
Wavefronts/iter:           16384 (elements/64)
----------------------------------------------------------------------
Expected counter values (per-wavefront, cumulative):
  SQ_INSTS_VALU_ADD_F64:         1720320
  SQ_INSTS_VALU_MUL_F64:         1720320
  SQ_INSTS_VALU_FMA_F64:               0
  Total FP64 VALU:               3440640
======================================================================

Running 100 iterations...

Performance:
  Time:      1.14 ms
  Bandwidth: 2213.89 GB/s
  GFLOP/s:   184.49

======================================================================
Kernel: Vector Add (ADD only)
----------------------------------------------------------------------
Array size:              1048576 elements
Iterations:                  100 (+ 5 warmup)
Wavefronts/iter:           16384 (elements/64)
----------------------------------------------------------------------
Expected counter values (per-wavefront, cumulative):
  SQ_INSTS_VALU_ADD_F64:         1720320
  SQ_INSTS_VALU_MUL_F64:               0
  SQ_INSTS_VALU_FMA_F64:               0
  Total FP64 VALU:               1720320
======================================================================

Running 100 iterations...

Performance:
  Time:      0.61 ms
  Bandwidth: 4133.05 GB/s
  GFLOP/s:   172.21

======================================================================
Kernel: Vector Multiply (MUL only)
----------------------------------------------------------------------
Array size:              1048576 elements
Iterations:                  100 (+ 5 warmup)
Wavefronts/iter:           16384 (elements/64)
----------------------------------------------------------------------
Expected counter values (per-wavefront, cumulative):
  SQ_INSTS_VALU_ADD_F64:               0
  SQ_INSTS_VALU_MUL_F64:         1720320
  SQ_INSTS_VALU_FMA_F64:               0
  Total FP64 VALU:               1720320
======================================================================

Running 100 iterations...

Performance:
  Time:      0.55 ms
  Bandwidth: 4587.67 GB/s
  GFLOP/s:   191.15

Verification:
  Expected a[0]: 2.00
  Actual a[0]:   2.00
  Status:        PASS

########################################################
#                    Test Complete                      #
########################################################

Waiting for all ranks to exit...

======================================================================
FP64 Monitor Analysis Summary
======================================================================

Counter Set:   fp64
Input files:      1
Total samples: 5
Active samples: 1
Idle samples:   4

----------------------------------------------------------------------
General Statistics
----------------------------------------------------------------------

  Total VALU:              72,253,440
  Total Instr:            196,116,480

----------------------------------------------------------------------
FP64 Instruction Counts
----------------------------------------------------------------------

  FP64 ADD:                 3,440,640
  FP64 MUL:                 3,440,640
  FP64 FMA:                 3,440,640
  FP64 Trans:                       0
  FP64 MFMA:                        0
  ----------------------------------------
  Total FP64 VALU:         10,321,920

FP64 %% of VALU:        14.29%
FP64 %% of Total:        5.26%

----------------------------------------------------------------------
Memory Usage
----------------------------------------------------------------------

CPU Resident Memory:
  HWM Min:        1475.7 MB
  HWM Max:        1475.7 MB
  HWM Mean:       1475.7 MB
  HWM StdDev:       0.00 MB
  HWM Total:        1.44 GB (1 files)

======================================================================

Report written to: fp64_monitor_output/fp64_report.txt
```

## Troubleshooting

- **Zeros:** Match `CMAKE_HIP_ARCHITECTURES` to the ASIC (`rocminfo` / `amd-smi`), confirm kernels run (e.g. `rocprofv3 --hip-trace`), and ensure ROCm is sourced.
- **“Failed to start context”:** The tool retries during startup; slow HSA init can trigger this briefly.
- **CMake cannot find rocprofiler-sdk:** `source` ROCm and/or `-DCMAKE_PREFIX_PATH="$ROCM_PATH"`.

More design detail: `IMPLEMENTATION_PLAN.md`.

## License

BSD 3-Clause — see `LICENSE`.
