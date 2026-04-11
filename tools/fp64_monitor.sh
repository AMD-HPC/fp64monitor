#!/bin/bash
#
# FP64 Monitor Wrapper Script
#
# This script wraps an application to enable FP64 instruction monitoring.
# Place this between mpirun and your executable:
#
#   mpirun -np 4 /path/to/fp64_monitor.sh ./your_app [args...]
#
# Command line options (override environment variables):
#   --interval MS         Sampling interval in ms (default: 1000)
#   --output PATTERN      Output file pattern (default: auto)
#   --format FORMAT       Output format: csv, json, text (default: csv)
#   --dir DIR             Output directory (default: fp64_monitor_output)
#   --gpu ID              GPU ID to monitor (default: 0)
#   --tmpdir DIR          Fast local storage for temp output (e.g., /dev/shm)
#   --use-rocmsmi         Enable rocm-smi memory monitoring
#   --counters TYPE       Counter set: fp64 (default), fp32, fp16, or int
#   --no-analyze          Disable auto-analysis after run
#   --verbose             Enable verbose output
#   --help                Show this help message
#
# Environment variables (optional, overridden by command line):
#   FP64_MONITOR_INTERVAL      - Sampling interval in ms (default: 1000)
#   FP64_MONITOR_OUTPUT        - Output file pattern (default: auto)
#   FP64_MONITOR_FORMAT        - Output format: csv, json, text (default: csv)
#   FP64_MONITOR_DIR           - Output directory (default: fp64_monitor_output)
#   FP64_MONITOR_GPU           - GPU ID to monitor (default: 0)
#   FP64_MONITOR_TMPDIR        - Fast local storage for temp output
#   FP64_MONITOR_USE_ROCMSMI   - Set to 1 to enable rocm-smi memory monitoring
#   FP64_MONITOR_ANALYZE       - Set to 0 to disable auto-analysis (default: 1)
#   FP64_MONITOR_VERBOSE       - Set to 1 for verbose output
#

set -e

# Show help message
show_help() {
    cat << 'EOF'
Usage: fp64_monitor.sh [options] -- <application> [app_args...]

Options:
  --interval MS         Sampling interval in ms (default: 1000)
  --output PATTERN      Output file pattern (default: auto)
  --format FORMAT       Output format: csv, json, text (default: csv)
  --dir DIR             Output directory (default: fp64_monitor_output)
  --gpu ID              GPU ID to monitor (default: 0)
  --tmpdir DIR          Fast local storage for temp output (e.g., /dev/shm)
  --use-rocmsmi         Enable rocm-smi memory monitoring
  --counters TYPE       Counter set: fp64 (default), fp32, fp16, or int
  --no-analyze          Disable auto-analysis after run
  --verbose             Enable verbose output
  --help                Show this help message

Examples:
  fp64_monitor.sh -- ./my_app
  fp64_monitor.sh --interval 500 --verbose -- ./my_app arg1 arg2
  mpirun -np 4 fp64_monitor.sh --use-rocmsmi --tmpdir /dev/shm -- ./my_mpi_app
EOF
    exit 0
}

# Parse command line arguments
APP_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --interval)
            export FP64_MONITOR_INTERVAL="$2"
            shift 2
            ;;
        --output)
            export FP64_MONITOR_OUTPUT="$2"
            shift 2
            ;;
        --format)
            export FP64_MONITOR_FORMAT="$2"
            shift 2
            ;;
        --dir)
            export FP64_MONITOR_DIR="$2"
            shift 2
            ;;
        --gpu)
            export FP64_MONITOR_GPU="$2"
            shift 2
            ;;
        --tmpdir)
            export FP64_MONITOR_TMPDIR="$2"
            shift 2
            ;;
        --use-rocmsmi)
            export FP64_MONITOR_USE_ROCMSMI=1
            shift
            ;;
        --counters)
            export FP64_MONITOR_COUNTERS="$2"
            shift 2
            ;;
        --no-analyze)
            export FP64_MONITOR_ANALYZE=0
            shift
            ;;
        --verbose)
            export FP64_MONITOR_VERBOSE=1
            shift
            ;;
        --help|-h)
            show_help
            ;;
        --)
            shift
            APP_ARGS+=("$@")
            break
            ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            echo "Or, did you forget \"--\" at the end of the fp64_monitor.sh arguments?" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
        *)
            APP_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set defaults for any unset variables
export FP64_MONITOR_INTERVAL="${FP64_MONITOR_INTERVAL:-1000}"

# Determine script directory and library path
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# Find library - expected to be in ../lib relative to script (install layout: bin/script, lib/library)
LIB_PATH="${SCRIPT_DIR}/../lib/libfp64_monitor_tool.so"

if [ ! -f "${LIB_PATH}" ]; then
    echo "Error: libfp64_monitor_tool.so not found at ${LIB_PATH}" >&2
    echo "Please ensure the tool is properly installed." >&2
    exit 1
fi

LIB_PATH="$(readlink -f "${LIB_PATH}")"

# Always use ROCP_TOOL_LIBRARIES to ensure rocprofiler runtime calls tool_init.
# The runtime will also call tool_fini, but we have a guard against double execution.
if [ -n "${ROCP_TOOL_LIBRARIES}" ]; then
    export ROCP_TOOL_LIBRARIES="${LIB_PATH}:${ROCP_TOOL_LIBRARIES}"
else
    export ROCP_TOOL_LIBRARIES="${LIB_PATH}"
fi

# When FP64_MONITOR_USE_ROCMSMI=1, additionally use LD_PRELOAD so our __libc_start_main
# wrapper can intercept main() exit and shut down before rocm-smi is torn down.
# Save original LD_PRELOAD to restore after application runs.
OLD_LD_PRELOAD="${LD_PRELOAD:-}"
if [ "${FP64_MONITOR_USE_ROCMSMI}" = "1" ]; then
    if [ -n "${LD_PRELOAD}" ]; then
        export LD_PRELOAD="${LIB_PATH}:${LD_PRELOAD}"
    else
        export LD_PRELOAD="${LIB_PATH}"
    fi
fi

# Detect MPI rank from environment
get_mpi_rank() {
    # Check various MPI implementations' rank environment variables
    if [ -n "${OMPI_COMM_WORLD_RANK}" ]; then
        echo "${OMPI_COMM_WORLD_RANK}"
    elif [ -n "${PMI_RANK}" ]; then
        echo "${PMI_RANK}"
    elif [ -n "${SLURM_PROCID}" ]; then
        echo "${SLURM_PROCID}"
    elif [ -n "${MV2_COMM_WORLD_RANK}" ]; then
        echo "${MV2_COMM_WORLD_RANK}"
    elif [ -n "${ALPS_APP_PE}" ]; then
        echo "${ALPS_APP_PE}"
    elif [ -n "${PALS_RANKID}" ]; then
        echo "${PALS_RANKID}"
    else
        echo "-1"
    fi
}

# Check if we have an application to run
if [ ${#APP_ARGS[@]} -eq 0 ]; then
    echo "Error: No application specified" >&2
    echo "Use --help for usage information" >&2
    exit 1
fi

# Set GPU from ROCR_VISIBLE_DEVICES
if [[ -v ROCR_VISIBLE_DEVICES ]] ; then
    export FP64_MONITOR_GPU=${ROCR_VISIBLE_DEVICES}
fi

# Run the application
"${APP_ARGS[@]}"
EXIT_CODE=$?

# Restore original LD_PRELOAD
export LD_PRELOAD="${OLD_LD_PRELOAD}"

# Get MPI rank
MPI_RANK=$(get_mpi_rank)

# Auto-analyze results (only on rank 0, after a delay)
ANALYZE="${FP64_MONITOR_ANALYZE:-1}"
if [ "${ANALYZE}" = "1" ] && [ "${MPI_RANK}" = "0" -o "${MPI_RANK}" = "-1" ]; then
    # Wait for other ranks to finish writing their files
    sleep 5
    
    # Find the analyze script
    ANALYZE_SCRIPT=""
    for path in "${SCRIPT_DIR}/analyze_results.py" \
                "${SCRIPT_DIR}/../bin/analyze_results.py" \
                "${SCRIPT_DIR}/../tests/analyze_results.py" \
                "${SCRIPT_DIR}/tests/analyze_results.py"; do
        if [ -f "${path}" ]; then
            ANALYZE_SCRIPT="${path}"
            break
        fi
    done
    
    # Find output files (check subdirectory first, then current directory for backwards compatibility)
    OUTPUT_DIR="${FP64_MONITOR_DIR:-fp64_monitor_output}"
    FILES_DIR=""
    if [ -d "${OUTPUT_DIR}" ]; then
        OUTPUT_FILES=$(ls "${OUTPUT_DIR}"/fp64_monitor_*.csv 2>/dev/null || true)
        if [ -n "${OUTPUT_FILES}" ]; then
            FILES_DIR="${OUTPUT_DIR}"
        fi
    fi
    # Fallback to current directory
    if [ -z "${OUTPUT_FILES}" ]; then
        OUTPUT_FILES=$(ls fp64_monitor_*.csv 2>/dev/null || true)
        if [ -n "${OUTPUT_FILES}" ]; then
            FILES_DIR="."
        fi
    fi
    
    if [ -n "${OUTPUT_FILES}" ] && [ -f "${ANALYZE_SCRIPT}" ]; then
        echo "" >&2
        # Pass all files to analysis script for aggregated output
        python3 "${ANALYZE_SCRIPT}" "${FILES_DIR}"/fp64_monitor_*.csv
    elif [ -z "${OUTPUT_FILES}" ]; then
        echo "Warning: No fp64_monitor_*.csv files found for analysis" >&2
    fi
fi

exit ${EXIT_CODE}
