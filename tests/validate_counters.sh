#!/bin/bash
#
# validate_counters.sh - Build and validate FP64 counter collection
#
# This script:
# 1. Builds the test program
# 2. Runs it with rocprofv3 to collect FP64 counters
# 3. Compares actual vs expected counter values
#
# Usage:
#   ./validate_counters.sh [array_size] [iterations] [kernel_type]
#
# Kernel types:
#   0 = Standard triad (likely FMA)
#   1 = Explicit FMA triad
#   2 = Separate MUL+ADD triad
#   3 = Vector add only
#   4 = Vector multiply only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
OUTPUT_DIR="${SCRIPT_DIR}/profiler_output"

# Default parameters
ARRAY_SIZE=${1:-1048576}  # 1M elements
ITERATIONS=${2:-100}
KERNEL_TYPE=${3:-1}       # Default to explicit FMA kernel

# Source ROCm environment if not already set
if [ -z "$ROCM_PATH" ]; then
    SOURCEME="/home/kevihuck/src/therock-tarball/sourceme-7.13.0a20260324-gfx94X.sh"
    if [ -f "$SOURCEME" ]; then
        echo "Sourcing ROCm environment..."
        source "$SOURCEME"
    else
        echo "ERROR: ROCM_PATH not set and sourceme script not found"
        exit 1
    fi
fi

echo "========================================================"
echo "FP64 Counter Validation Script"
echo "========================================================"
echo ""
echo "Configuration:"
echo "  ROCM_PATH:    $ROCM_PATH"
echo "  Array size:   $ARRAY_SIZE elements"
echo "  Iterations:   $ITERATIONS"
echo "  Kernel type:  $KERNEL_TYPE"
echo ""

# Build test program
echo "--------------------------------------------------------"
echo "Building test program..."
echo "--------------------------------------------------------"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo ""
echo "Build complete: $BUILD_DIR/stream_triad_fp64"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"/*

# Define counters to collect
COUNTERS="SQ_INSTS SQ_INSTS_VALU SQ_INSTS_VALU_ADD_F64 SQ_INSTS_VALU_MUL_F64 SQ_INSTS_VALU_FMA_F64 SQ_INSTS_VALU_TRANS_F64"

echo "--------------------------------------------------------"
echo "Running with rocprofv3..."
echo "--------------------------------------------------------"
echo "Counters: $COUNTERS"
echo ""

# Run with rocprofv3
cd "$OUTPUT_DIR"
rocprofv3 --pmc $COUNTERS \
    -o fp64_counters \
    -f csv \
    -- "$BUILD_DIR/stream_triad_fp64" $ARRAY_SIZE $ITERATIONS $KERNEL_TYPE

echo ""
echo "--------------------------------------------------------"
echo "Profiler Output Files:"
echo "--------------------------------------------------------"
ls -la "$OUTPUT_DIR"

echo ""
echo "--------------------------------------------------------"
echo "Counter Results:"
echo "--------------------------------------------------------"

# Find the counter CSV file
COUNTER_FILE=$(find "$OUTPUT_DIR" -name "*counter*.csv" | head -1)

if [ -f "$COUNTER_FILE" ]; then
    echo "Counter file: $COUNTER_FILE"
    echo ""
    
    # Display the results
    echo "Raw counter data:"
    cat "$COUNTER_FILE"
    echo ""
    
    # Calculate expected values
    TOTAL_ELEMENTS=$((ARRAY_SIZE * ITERATIONS))
    
    echo "--------------------------------------------------------"
    echo "Expected vs Actual Comparison:"
    echo "--------------------------------------------------------"
    echo ""
    
    case $KERNEL_TYPE in
        0|1)
            EXPECTED_FMA=$TOTAL_ELEMENTS
            EXPECTED_ADD=0
            EXPECTED_MUL=0
            echo "Kernel: FMA-based triad"
            echo "Expected SQ_INSTS_VALU_FMA_F64: $EXPECTED_FMA"
            echo "Expected SQ_INSTS_VALU_ADD_F64: $EXPECTED_ADD"
            echo "Expected SQ_INSTS_VALU_MUL_F64: $EXPECTED_MUL"
            ;;
        2)
            EXPECTED_FMA=0
            EXPECTED_ADD=$TOTAL_ELEMENTS
            EXPECTED_MUL=$TOTAL_ELEMENTS
            echo "Kernel: Separate MUL+ADD triad"
            echo "Expected SQ_INSTS_VALU_FMA_F64: $EXPECTED_FMA"
            echo "Expected SQ_INSTS_VALU_ADD_F64: $EXPECTED_ADD"
            echo "Expected SQ_INSTS_VALU_MUL_F64: $EXPECTED_MUL"
            ;;
        3)
            EXPECTED_FMA=0
            EXPECTED_ADD=$TOTAL_ELEMENTS
            EXPECTED_MUL=0
            echo "Kernel: Vector Add only"
            echo "Expected SQ_INSTS_VALU_ADD_F64: $EXPECTED_ADD"
            ;;
        4)
            EXPECTED_FMA=0
            EXPECTED_ADD=0
            EXPECTED_MUL=$TOTAL_ELEMENTS
            echo "Kernel: Vector Multiply only"
            echo "Expected SQ_INSTS_VALU_MUL_F64: $EXPECTED_MUL"
            ;;
    esac
    
    echo ""
    echo "Total expected FP64 elements processed: $TOTAL_ELEMENTS"
    echo ""
    
    # Try to parse and compare (basic grep-based extraction)
    echo "Extracting actual values from CSV..."
    
    # Sum up counter values across all kernel invocations
    # The CSV has columns like: Counter_Name,Counter_Value,...
    
    if grep -q "SQ_INSTS_VALU_FMA_F64" "$COUNTER_FILE" 2>/dev/null; then
        ACTUAL_FMA=$(grep "SQ_INSTS_VALU_FMA_F64" "$COUNTER_FILE" | awk -F',' '{sum += $NF} END {print sum}')
        echo "Actual SQ_INSTS_VALU_FMA_F64: $ACTUAL_FMA"
    fi
    
    if grep -q "SQ_INSTS_VALU_ADD_F64" "$COUNTER_FILE" 2>/dev/null; then
        ACTUAL_ADD=$(grep "SQ_INSTS_VALU_ADD_F64" "$COUNTER_FILE" | awk -F',' '{sum += $NF} END {print sum}')
        echo "Actual SQ_INSTS_VALU_ADD_F64: $ACTUAL_ADD"
    fi
    
    if grep -q "SQ_INSTS_VALU_MUL_F64" "$COUNTER_FILE" 2>/dev/null; then
        ACTUAL_MUL=$(grep "SQ_INSTS_VALU_MUL_F64" "$COUNTER_FILE" | awk -F',' '{sum += $NF} END {print sum}')
        echo "Actual SQ_INSTS_VALU_MUL_F64: $ACTUAL_MUL"
    fi
    
else
    echo "WARNING: Counter CSV file not found!"
    echo "Available files in output directory:"
    ls -la "$OUTPUT_DIR"
fi

echo ""
echo "========================================================"
echo "Validation Complete"
echo "========================================================"
echo ""
echo "Note: Counter values may differ from expectations due to:"
echo "  - Compiler optimizations (FMA fusion)"
echo "  - Thread divergence handling"
echo "  - Warmup iterations (included in counter totals)"
echo "  - Per-XCC/SE dimension reporting"
echo ""
echo "For accurate comparison, sum all instances of each counter"
echo "and compare against (array_size * iterations)."
