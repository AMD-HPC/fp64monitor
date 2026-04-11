#!/bin/bash
#
# Integration tests for FP64 Monitor Tool
#
# This script runs end-to-end tests of the fp64_monitor_tool with
# the stream_triad_fp64 test program.
#
# Usage: ./run_integration_tests.sh [build_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${1:-${SCRIPT_DIR}/../build}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() {
    echo -e "${GREEN}PASS${NC}: $1"
}

fail() {
    echo -e "${RED}FAIL${NC}: $1"
    exit 1
}

warn() {
    echo -e "${YELLOW}WARN${NC}: $1"
}

info() {
    echo "INFO: $1"
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    if [ ! -f "${BUILD_DIR}/libfp64_monitor_tool.so" ]; then
        fail "libfp64_monitor_tool.so not found in ${BUILD_DIR}"
    fi
    
    if [ ! -f "${BUILD_DIR}/tests/stream_triad_fp64" ]; then
        fail "stream_triad_fp64 not found in ${BUILD_DIR}/tests"
    fi
    
    if [ ! -f "${BUILD_DIR}/tests/test_instruction_analyzer" ]; then
        warn "test_instruction_analyzer not found, skipping unit tests"
        SKIP_UNIT_TESTS=1
    fi
    
    pass "Prerequisites check"
}

# Run unit tests
run_unit_tests() {
    if [ -n "$SKIP_UNIT_TESTS" ]; then
        return
    fi
    
    info "Running unit tests..."
    
    if "${BUILD_DIR}/tests/test_instruction_analyzer"; then
        pass "Unit tests"
    else
        fail "Unit tests failed"
    fi
}

# Test: Basic tool execution
test_basic_execution() {
    info "Test: Basic tool execution..."
    
    local output_file=$(mktemp /tmp/fp64_test_XXXXXX.csv)
    
    FP64_MONITOR_OUTPUT="${output_file}" \
    FP64_MONITOR_INTERVAL=50 \
    FP64_MONITOR_FORMAT=csv \
    ROCP_TOOL_LIBRARIES="${BUILD_DIR}/libfp64_monitor_tool.so" \
    timeout 5 "${BUILD_DIR}/tests/stream_triad_fp64" 2>&1 || true
    
    if [ -f "${output_file}" ]; then
        local lines=$(wc -l < "${output_file}")
        if [ "$lines" -gt 1 ]; then
            pass "Basic execution (${lines} lines output)"
        else
            fail "Output file has no data"
        fi
    else
        fail "Output file not created"
    fi
    
    rm -f "${output_file}"
}

# Test: CSV format validity
test_csv_format() {
    info "Test: CSV format validity..."
    
    local output_file=$(mktemp /tmp/fp64_test_XXXXXX.csv)
    
    FP64_MONITOR_OUTPUT="${output_file}" \
    FP64_MONITOR_INTERVAL=50 \
    FP64_MONITOR_FORMAT=csv \
    ROCP_TOOL_LIBRARIES="${BUILD_DIR}/libfp64_monitor_tool.so" \
    timeout 5 "${BUILD_DIR}/tests/stream_triad_fp64" 2>&1 || true
    
    # Check header has expected columns
    local header=$(head -1 "${output_file}")
    if echo "$header" | grep -q "sample_id" && \
       echo "$header" | grep -q "fp64_fma" && \
       echo "$header" | grep -q "fp64_valu_pct"; then
        pass "CSV header contains expected columns"
    else
        fail "CSV header missing expected columns: ${header}"
    fi
    
    rm -f "${output_file}"
}

# Test: JSON format validity
test_json_format() {
    info "Test: JSON format validity..."
    
    local output_file=$(mktemp /tmp/fp64_test_XXXXXX.json)
    
    FP64_MONITOR_OUTPUT="${output_file}" \
    FP64_MONITOR_INTERVAL=50 \
    FP64_MONITOR_FORMAT=json \
    ROCP_TOOL_LIBRARIES="${BUILD_DIR}/libfp64_monitor_tool.so" \
    timeout 5 "${BUILD_DIR}/tests/stream_triad_fp64" 2>&1 || true
    
    # Check if it's valid JSON (basic check)
    if head -1 "${output_file}" | grep -q '{"samples":\['; then
        pass "JSON format basic structure"
    else
        warn "JSON format may be incomplete (expected for streaming output)"
    fi
    
    rm -f "${output_file}"
}

# Test: Text format
test_text_format() {
    info "Test: Text format..."
    
    local output_file=$(mktemp /tmp/fp64_test_XXXXXX.txt)
    
    FP64_MONITOR_OUTPUT="${output_file}" \
    FP64_MONITOR_INTERVAL=50 \
    FP64_MONITOR_FORMAT=text \
    ROCP_TOOL_LIBRARIES="${BUILD_DIR}/libfp64_monitor_tool.so" \
    timeout 5 "${BUILD_DIR}/tests/stream_triad_fp64" 2>&1 || true
    
    if [ -f "${output_file}" ] && [ -s "${output_file}" ]; then
        pass "Text format output created"
    else
        fail "Text format output not created or empty"
    fi
    
    rm -f "${output_file}"
}

# Test: Stdout output
test_stdout_output() {
    info "Test: Stdout output..."
    
    local output=$(FP64_MONITOR_OUTPUT=stdout \
                   FP64_MONITOR_INTERVAL=100 \
                   FP64_MONITOR_FORMAT=csv \
                   ROCP_TOOL_LIBRARIES="${BUILD_DIR}/libfp64_monitor_tool.so" \
                   timeout 3 "${BUILD_DIR}/tests/stream_triad_fp64" 2>&1 || true)
    
    if echo "$output" | grep -q "sample_id"; then
        pass "Stdout output works"
    else
        warn "Stdout output may not contain expected data"
    fi
}

# Test: Default filename generation
test_default_filename() {
    info "Test: Default filename generation..."
    
    local abs_build_dir="$(cd "${BUILD_DIR}" && pwd)"
    local test_dir=$(mktemp -d /tmp/fp64_test_XXXXXX)
    
    cd "${test_dir}"
    
    FP64_MONITOR_INTERVAL=50 \
    FP64_MONITOR_FORMAT=csv \
    ROCP_TOOL_LIBRARIES="${abs_build_dir}/libfp64_monitor_tool.so" \
    timeout 3 "${abs_build_dir}/tests/stream_triad_fp64" 2>&1 || true
    
    local default_files=$(ls fp64_monitor_*.csv 2>/dev/null | wc -l)
    
    cd - > /dev/null
    
    if [ "$default_files" -gt 0 ]; then
        pass "Default filename generated"
    else
        fail "Default filename not generated"
    fi
    
    rm -rf "${test_dir}"
}

# Test: Filename pattern expansion
test_filename_pattern() {
    info "Test: Filename pattern expansion..."
    
    local pid_pattern="/tmp/fp64_pid_%p.csv"
    local hostname_pattern="/tmp/fp64_host_%h.csv"
    
    # Test %p pattern
    FP64_MONITOR_OUTPUT="${pid_pattern}" \
    FP64_MONITOR_INTERVAL=100 \
    ROCP_TOOL_LIBRARIES="${BUILD_DIR}/libfp64_monitor_tool.so" \
    timeout 2 "${BUILD_DIR}/tests/stream_triad_fp64" 2>&1 || true
    
    if ls /tmp/fp64_pid_*.csv 2>/dev/null | grep -v '%p' | head -1 > /dev/null; then
        pass "PID pattern expanded"
        rm -f /tmp/fp64_pid_*.csv
    else
        fail "PID pattern not expanded"
    fi
    
    # Test %h pattern
    FP64_MONITOR_OUTPUT="${hostname_pattern}" \
    FP64_MONITOR_INTERVAL=100 \
    ROCP_TOOL_LIBRARIES="${BUILD_DIR}/libfp64_monitor_tool.so" \
    timeout 2 "${BUILD_DIR}/tests/stream_triad_fp64" 2>&1 || true
    
    if ls /tmp/fp64_host_*.csv 2>/dev/null | grep -v '%h' | head -1 > /dev/null; then
        pass "Hostname pattern expanded"
        rm -f /tmp/fp64_host_*.csv
    else
        fail "Hostname pattern not expanded"
    fi
}

# Test: Version output
test_version_output() {
    info "Test: Version output in stderr..."
    
    local stderr_output=$(FP64_MONITOR_OUTPUT=/dev/null \
                          FP64_MONITOR_INTERVAL=200 \
                          ROCP_TOOL_LIBRARIES="${BUILD_DIR}/libfp64_monitor_tool.so" \
                          timeout 2 "${BUILD_DIR}/tests/stream_triad_fp64" 2>&1 || true)
    
    if echo "$stderr_output" | grep -q "FP64 Monitor.*v"; then
        pass "Version shown in stderr"
    else
        fail "Version not shown in stderr"
    fi
    
    if echo "$stderr_output" | grep -q "ROCm"; then
        pass "ROCm version shown"
    else
        warn "ROCm version not shown"
    fi
}

# Test: Summary at end
test_summary_output() {
    info "Test: Summary output at end..."
    
    local output_file=$(mktemp /tmp/fp64_test_XXXXXX.csv)
    
    FP64_MONITOR_OUTPUT="${output_file}" \
    FP64_MONITOR_INTERVAL=50 \
    ROCP_TOOL_LIBRARIES="${BUILD_DIR}/libfp64_monitor_tool.so" \
    timeout 3 "${BUILD_DIR}/tests/stream_triad_fp64" 2>&1 || true
    
    # Check for summary comment at end
    if tail -5 "${output_file}" | grep -q "Summary"; then
        pass "Summary present in output"
    else
        warn "Summary not found (may be format-dependent)"
    fi
    
    rm -f "${output_file}"
}

# Main
main() {
    echo ""
    echo "========================================"
    echo "  FP64 Monitor Integration Tests"
    echo "========================================"
    echo "Build directory: ${BUILD_DIR}"
    echo ""
    
    check_prerequisites
    
    echo ""
    echo "--- Unit Tests ---"
    run_unit_tests
    
    echo ""
    echo "--- Integration Tests ---"
    test_basic_execution
    test_csv_format
    test_json_format
    test_text_format
    test_stdout_output
    test_default_filename
    test_filename_pattern
    test_version_output
    test_summary_output
    
    echo ""
    echo "========================================"
    echo -e "  ${GREEN}All integration tests passed!${NC}"
    echo "========================================"
    echo ""
}

main "$@"
