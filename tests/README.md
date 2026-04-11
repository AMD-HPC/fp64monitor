# Test Programs for FP64 Counter Validation

This directory contains test programs with known FP64 instruction counts for validating the measure_flops utility.

## Test Programs

### stream_triad_fp64.hip

A STREAM Triad benchmark using FP64 (double) arrays. The triad operation is:

```
a[i] = b[i] + scalar * c[i]
```

This program provides several kernel variants with predictable FP64 instruction patterns:

| Kernel | Type | Expected FP64 Instructions per Element |
|--------|------|----------------------------------------|
| 0 | Standard Triad | 1 FMA (compiler may optimize) |
| 1 | Explicit FMA | 1 FMA |
| 2 | Separate MUL+ADD | 1 MUL + 1 ADD |
| 3 | Vector Add | 1 ADD |
| 4 | Vector Multiply | 1 MUL |

#### Building

```bash
# Source ROCm environment
source /home/kevihuck/src/therock-tarball/sourceme-7.13.0a20260324-gfx94X.sh

# Build
cd tests
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$ROCM_PATH
make
```

#### Usage

```bash
./stream_triad_fp64 [array_size] [iterations] [kernel_type]

# Examples:
./stream_triad_fp64                     # Default: 1M elements, 100 iterations, all kernels
./stream_triad_fp64 1048576 100 1       # 1M elements, 100 iters, explicit FMA kernel
./stream_triad_fp64 10000000 50 3       # 10M elements, 50 iters, vector add kernel
```

#### Expected Counter Values

For array size `N` and `I` iterations:

| Kernel | SQ_INSTS_VALU_ADD_F64 | SQ_INSTS_VALU_MUL_F64 | SQ_INSTS_VALU_FMA_F64 |
|--------|----------------------|----------------------|----------------------|
| 0 (Standard) | 0 | 0 | N × I |
| 1 (FMA) | 0 | 0 | N × I |
| 2 (Separate) | N × I | N × I | 0 |
| 3 (Add) | N × I | 0 | 0 |
| 4 (Mul) | 0 | N × I | 0 |

**Note**: Actual values may include warmup iterations and vary slightly due to thread divergence handling.

## Validation Script

### validate_counters.sh

Automated script to build, run with rocprofv3, and compare expected vs actual counter values.

```bash
# Run validation
./validate_counters.sh [array_size] [iterations] [kernel_type]

# Examples:
./validate_counters.sh                   # Default settings
./validate_counters.sh 1048576 100 1     # 1M elements, 100 iters, FMA kernel
```

The script will:
1. Build the test program
2. Run it with rocprofv3 to collect FP64 counters
3. Display expected vs actual values

## Profiling with rocprofv3

### Manual Profiling

```bash
# Collect FP64 counters
rocprofv3 --pmc "SQ_INSTS SQ_INSTS_VALU SQ_INSTS_VALU_ADD_F64 SQ_INSTS_VALU_MUL_F64 SQ_INSTS_VALU_FMA_F64" \
    -o results -f csv \
    -- ./stream_triad_fp64 1048576 100 1

# View results
cat results/*/counter_collection.csv
```

### Full Instruction Mix

```bash
rocprofv3 --pmc "SQ_INSTS SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM \
    SQ_INSTS_VALU_ADD_F64 SQ_INSTS_VALU_MUL_F64 SQ_INSTS_VALU_FMA_F64 SQ_INSTS_VALU_TRANS_F64 \
    SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32 SQ_INSTS_VALU_FMA_F32" \
    -o results -f csv \
    -- ./stream_triad_fp64 1048576 100 1
```

## Interpreting Results

### Important: Counter Semantics

**SQ_INSTS counters count instructions issued to the sequencer, which is per-wavefront (64 threads), NOT per-thread.**

For example, with `array_size=1,048,576`:
- Wavefronts needed: 1,048,576 ÷ 64 = **16,384**
- FMA instructions per wavefront: 1
- Expected `SQ_INSTS_VALU_FMA_F64` per dispatch: **16,384**

This is verified by actual profiler output:
```
Per-dispatch values:
  SQ_INSTS_VALU_FMA_F64: 16,384  ✓ (matches wavefront count)
  SQ_INSTS_VALU:         131,072 (includes other VALU ops)
  SQ_INSTS:              360,448 (all instruction types)
```

### Counter Dimensions

On MI300X and similar GPUs, counters may be reported per hardware unit (XCC, Shader Engine, etc.). 
With dispatch profiling (rocprofv3), values are typically aggregated per kernel dispatch.

### FP64 Percentage Calculation

The FP64 percentage of VALU will be less than 100% even for pure FP64 kernels because:
- Address calculations generate integer VALU instructions
- Boundary checks (if idx < n) generate comparison instructions
- Loop control may add additional instructions

For the STREAM Triad FMA kernel:
```
FP64 % of VALU = 16,384 / 131,072 = 12.5%
```

This 12.5% represents the actual FP64 compute portion of the kernel's VALU workload.

### Validation Example

Actual profiler output for `array_size=1,048,576`, `iterations=10` (+ 5 warmup = 15 dispatches):

| Counter | Per Dispatch | Total (15 dispatches) |
|---------|--------------|----------------------|
| SQ_INSTS_VALU_FMA_F64 | 16,384 | 245,760 |
| SQ_INSTS_VALU | 131,072 | 1,966,080 |
| SQ_INSTS | 360,448 | 5,406,720 |

### Discrepancies

Counter values may differ from naive calculations due to:

1. **Wavefront counting**: Counters are per-wavefront, not per-thread
2. **Warmup iterations**: The test includes 5 warmup iterations
3. **Additional instructions**: Address calc, bounds checks, etc.
4. **Compiler optimizations**: May generate different instruction sequences
