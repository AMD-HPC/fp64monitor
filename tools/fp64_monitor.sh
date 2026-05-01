#!/bin/bash
#
# FP64 Monitor Wrapper Script
#
# This script wraps an application to enable GPU instruction monitoring using
# hardware performance counters. It supports various counter sets for analyzing
# different aspects of GPU workloads.
#
# Usage:
#   fp64_monitor.sh [options] -- <application> [app_args...]
#   mpirun -np 4 fp64_monitor.sh [options] -- ./your_mpi_app [args...]
#
# For detailed help: fp64_monitor.sh --fullhelp
#

# Use patched nohwloc libfabric instead of /lib64 version to prevent exit crash -JCG 5/1/26
export LD_LIBRARY_PATH="/usr/tce/packages/libfabric/libfabric-2.3.1nohwloc/lib64:${LD_LIBRARY_PATH}"

ROCM_VERSION_STRING=@ROCM_VERSION_STRING@
VERSION_PROVIDED=0
INSTALL_PREFIX=@CMAKE_INSTALL_PREFIX@
OUTPUT_DIR_POSTFIX=""
if type -p flux &> /dev/null; then
    jid=`flux getattr jobid`
    OUTPUT_DIR_POSTFIX="_${jid}"
    OUTPUT_DIR="${FP64_MONITOR_DIR:-fp64_monitor_output}${OUTPUT_DIR_POSTFIX}"
    export FP64_MONITOR_DIR=${OUTPUT_DIR}
fi

set -e

# Show brief help message (command line arguments only)
show_help() {
    cat << 'EOF'
FP64 Monitor - GPU Instruction Profiling Tool

Usage: fp64_monitor.sh [options] -- <application> [app_args...]

Options:
  --interval MS         Sampling interval in milliseconds (default: 1000)
  --output PATTERN      Output file pattern (default: auto-generated)
  --format FORMAT       Output format: csv, json, or text (default: csv)
  --rocm VERSION        ROCm version (default: @ROCM_VERSION_STRING@)
  --dir DIR             Output directory (default: fp64_monitor_output)
  --gpu ID              GPU ID to monitor (default: 0)
  --counters TYPE       Counter set: fp64, fp32, fp16, int, salu, highlevel, highlevel2
  --use-amdsmi          Enable GPU memory monitoring via AMD SMI
  --tmpdir DIR          Fast local storage for temp output (e.g., /dev/shm)
  --preload             Use LD_PRELOAD wrapper for clean shutdown
  --no-analyze          Skip automatic analysis after run
  --verbose             Enable verbose output
  --help                Show this help message
  --fullhelp            Show detailed help with counter descriptions and examples

EOF
    exit 0
}

# Show full help message with all details
show_fullhelp() {
    cat << 'EOF'
FP64 Monitor - GPU Instruction Profiling Tool

Usage: fp64_monitor.sh [options] -- <application> [app_args...]

GENERAL OPTIONS:
  --interval MS         Sampling interval in milliseconds (default: 1000)
  --output PATTERN      Output file pattern (default: auto-generated)
  --format FORMAT       Output format: csv, json, or text (default: csv)
  --rocm VERSION        ROCm version (default: @ROCM_VERSION_STRING@)
  --dir DIR             Output directory (default: fp64_monitor_output)
  --gpu ID              GPU ID to monitor (default: 0, or ROCR_VISIBLE_DEVICES)
  --verbose             Enable verbose output and rocprofiler-sdk logging
  --help                Show brief help message
  --fullhelp            Show this detailed help message

COUNTER SETS (--counters TYPE):
  fp64        FP64 instruction breakdown (default)
              Counters: SQ_INSTS, SQ_INSTS_VALU, SQ_INSTS_VALU_ADD_F64,
                        SQ_INSTS_VALU_MUL_F64, SQ_INSTS_VALU_FMA_F64,
                        SQ_INSTS_VALU_TRANS_F64, SQ_INSTS_MFMA, SQ_INSTS_MFMA_MOPS_F64

  fp32        FP32 instruction breakdown
              Counters: SQ_INSTS, SQ_INSTS_VALU, SQ_INSTS_VALU_ADD_F32,
                        SQ_INSTS_VALU_MUL_F32, SQ_INSTS_VALU_FMA_F32,
                        SQ_INSTS_VALU_TRANS_F32, SQ_INSTS_MFMA, SQ_INSTS_MFMA_MOPS_F32

  fp16        FP16/BF16 instruction breakdown
              Counters: SQ_INSTS, SQ_INSTS_VALU, SQ_INSTS_VALU_ADD_F16,
                        SQ_INSTS_VALU_MUL_F16, SQ_INSTS_VALU_FMA_F16,
                        SQ_INSTS_MFMA, SQ_INSTS_MFMA_MOPS_F16, SQ_INSTS_MFMA_MOPS_BF16

  int         Integer instruction breakdown
              Counters: SQ_INSTS, SQ_INSTS_VALU, SQ_INSTS_VALU_INT32,
                        SQ_INSTS_VALU_INT64, SQ_INSTS_MFMA, SQ_INSTS_MFMA_MOPS_I8

  salu        Scalar ALU and memory breakdown
              Counters: SQ_INSTS, SQ_INSTS_SALU, SQ_INSTS_SMEM, SQ_INSTS_FLAT,
                        SQ_INSTS_LDS, SQ_INSTS_BRANCH

  highlevel   High-level instruction mix (7 categories)
              Counters: SQ_INSTS, SQ_INSTS_VALU, SQ_INSTS_SALU, SQ_INSTS_VMEM,
                        SQ_INSTS_SMEM, SQ_INSTS_LDS, SQ_INSTS_BRANCH, SQ_INSTS_MFMA

  highlevel2  Other instruction types
              Counters: SQ_INSTS, SQ_INSTS_FLAT, SQ_INSTS_GDS, SQ_INSTS_EXP_GDS,
                        SQ_INSTS_SENDMSG, SQ_INSTS_VSKIPPED

MEMORY MONITORING:
  --use-amdsmi          Enable GPU memory monitoring via AMD SMI
                        Reports VRAM usage, GTT usage, and high-water marks
                        Automatically enables --preload for clean shutdown

ADVANCED OPTIONS:
  --tmpdir DIR          Write samples to fast local storage (e.g., /dev/shm),
                        then copy to final destination at exit. Useful for
                        reducing I/O load on parallel filesystems.
  --preload             Use LD_PRELOAD to wrap main() for clean shutdown.
                        Required for --use-amdsmi, optional otherwise.
  --no-analyze          Skip automatic analysis after run completes

ENVIRONMENT VARIABLES (overridden by command line options):
  FP64_MONITOR_INTERVAL      Sampling interval in ms (default: 1000)
  FP64_MONITOR_OUTPUT        Output file pattern
  FP64_MONITOR_FORMAT        Output format: csv, json, text
  FP64_MONITOR_DIR           Output directory
  FP64_MONITOR_GPU           GPU ID to monitor
  FP64_MONITOR_COUNTERS      Counter set name
  FP64_MONITOR_TMPDIR        Temporary storage directory
  FP64_MONITOR_USE_AMDSMI    Set to 1 to enable AMD SMI memory monitoring
  FP64_MONITOR_PRELOAD       Set to 1 to enable LD_PRELOAD wrapper
  FP64_MONITOR_ANALYZE       Set to 0 to disable auto-analysis
  FP64_MONITOR_VERBOSE       Set to 1 for verbose output

EXAMPLES:
  # Basic FP64 profiling
  fp64_monitor.sh -- ./my_app

  # Profile with memory monitoring and 500ms sampling
  fp64_monitor.sh --interval 500 --use-amdsmi -- ./my_app

  # High-level instruction mix analysis
  fp64_monitor.sh --counters highlevel --use-amdsmi -- ./my_app

  # MPI application with temp storage
  mpirun -np 4 fp64_monitor.sh --use-amdsmi --tmpdir /dev/shm -- ./my_mpi_app

  # Verbose output for debugging
  fp64_monitor.sh --verbose --counters fp64 -- ./my_app

OUTPUT:
  CSV files are written to the output directory (default: fp64_monitor_output/).
  After the run completes, an automatic analysis is performed showing:
  - Instruction counts aggregated across all MPI ranks
  - Percentages relative to total instructions
  - Memory high-water marks (if --use-amdsmi is used)
  - A detailed report is saved to fp64_report.txt

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
        --rocm)
            export ROCM_VERSION_STRING="$2"
            export VERSION_PROVIDED=1
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
        --use-amdsmi)
            export FP64_MONITOR_USE_AMDSMI=1
            export FP64_MONITOR_PRELOAD=1
            shift
            ;;
        --preload)
            export FP64_MONITOR_PRELOAD=1
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
        --fullhelp)
            show_fullhelp
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

# Set defaults for any unset variables
export FP64_MONITOR_INTERVAL="${FP64_MONITOR_INTERVAL:-1000}"

# Determine script directory and library path
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# Get MPI rank
MPI_RANK=$(get_mpi_rank)

if [ $VERSION_PROVIDED -eq 0 ] ; then
    failed=0
    SCRIPT_NAME=$(basename "$(readlink -f "${BASH_SOURCE[0]}")")
    # Attempt to identify the executable requirements
    executable=$(echo $APP_ARGS | awk '{print $1}')
    # Grep will set an error condition on failure, so don't exit :D
    set +e
    executable_type=$(file $executable | grep -c "ELF")
    set -e
    if [ $executable_type -eq 1 ] ; then
        FILENAME=$(ldd $executable | grep libhsa-runtime | head -n 1 | awk '{print $3}')
        version=$(echo "$FILENAME" | sed -r 's/.*rocm-([0-9.]+)\/.*/\1/')
        MAJOR="${version%%.*}"
        if [ "$MAJOR" -eq "7" ] ; then
            ROCM_VERSION_STRING=7.2.1
        elif [ "$MAJOR" -eq "6" ] ; then
            ROCM_VERSION_STRING=6.4.3
        else
            failed=1
        fi
    else
        failed=1
    fi
    if [ "${MPI_RANK}" = "0" -o "${MPI_RANK}" = "-1" ] ; then
        if [ $failed -eq 1 ] ; then
            echo "$SCRIPT_NAME: Unable to detect ROCm version from executable, using $ROCM_VERSION_STRING"
        else
            echo "$SCRIPT_NAME: Detected ROCm major version $MAJOR, using tool version $ROCM_VERSION_STRING"
        fi
    fi
fi

# Find library - expected to be in ../lib relative to script (install layout: bin/script, lib/library).
# Shared library from CMake target fp64_monitor_tool_${ROCM_VERSION_STRING} -> libfp64_monitor_tool_${ROCM_VERSION_STRING}.so
LIB_PATH="${SCRIPT_DIR}/../lib/libfp64_monitor_tool_${ROCM_VERSION_STRING}.so"

if [ ! -f "${LIB_PATH}" ]; then
    echo "Error: libfp64_monitor_tool_${ROCM_VERSION_STRING}.so not found at ${LIB_PATH}" >&2
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

# When FP64_MONITOR_USE_AMDSMI=1 or FP64_MONITOR_PRELOAD=1, use LD_PRELOAD so our
# __libc_start_main wrapper can intercept main() exit and shut down cleanly.
# This prevents possible race conditions when amd-smi or other libraries are torn down.
# Save original LD_PRELOAD to restore after application runs.
OLD_LD_PRELOAD="${LD_PRELOAD:-}"
if [ "${FP64_MONITOR_PRELOAD}" = "1" ]; then
    if [ -n "${LD_PRELOAD}" ]; then
        export LD_PRELOAD="${LIB_PATH}:${LD_PRELOAD}"
    else
        export LD_PRELOAD="${LIB_PATH}"
    fi
fi

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

# Suppress verbose rocprofiler-sdk logging by default
# GLOG_minloglevel: 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
# Only show these messages if --verbose is used
if [ "${FP64_MONITOR_VERBOSE}" != "1" ]; then
    export GLOG_minloglevel=3
fi

# Run the application
"${APP_ARGS[@]}"
EXIT_CODE=$?

# Restore original LD_PRELOAD
export LD_PRELOAD="${OLD_LD_PRELOAD}"

# Auto-analyze results (only on rank 0, after a delay)
ANALYZE="${FP64_MONITOR_ANALYZE:-1}"
if [ "${ANALYZE}" = "1" ] && [ "${MPI_RANK}" = "0" -o "${MPI_RANK}" = "-1" ]; then
    # Wait for other ranks to finish writing their files
    echo "Waiting for all ranks to exit..."
    sleep 5
    
    # Find the analyze script
    ANALYZE_SCRIPT=""
    for path in "${SCRIPT_DIR}/fp64_monitor_analyze_results.py" \
                "${SCRIPT_DIR}/../bin/fp64_monitor_analyze_results.py" \
                "${SCRIPT_DIR}/../tests/fp64_monitor_analyze_results.py" \
                "${SCRIPT_DIR}/tests/fp64_monitor_analyze_results.py"; do
        if [ -f "${path}" ]; then
            ANALYZE_SCRIPT="${path}"
            break
        fi
    done
    
    # Find output files (check subdirectory first, then current directory for backwards compatibility)
    FILES_DIR=""
    if [ -d "${FP64_MONITOR_DIR}" ]; then
        OUTPUT_FILES=$(ls "${FP64_MONITOR_DIR}"/fp64_monitor_*.csv 2>/dev/null || true)
        if [ -n "${OUTPUT_FILES}" ]; then
            FILES_DIR="${FP64_MONITOR_DIR}"
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
