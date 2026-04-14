#!/usr/bin/env python3
"""
Validate fp64_monitor counter collection against rocprofv3.

This script compares counter values collected by fp64_monitor against
rocprofv3's --pmc counter collection to verify accuracy.

Usage:
    python3 validate_counters.py <fp64_monitor_csv> <rocprofv3_csv>
    
Example:
    python3 validate_counters.py fp64_monitor_output/*.csv rocprof_output/*/*.csv

The script:
1. Parses fp64_monitor CSV (takes final cumulative value)
2. Parses rocprofv3 CSV (sums per-dispatch values)
3. Compares corresponding counters
4. Reports differences
"""

import sys
import csv
import glob
from pathlib import Path
from collections import defaultdict


# Mapping between fp64_monitor column names and rocprofv3 counter names
COUNTER_MAP = {
    # fp64_monitor column -> rocprofv3 counter name
    'total_instructions': 'SQ_INSTS',
    'valu_instructions': 'SQ_INSTS_VALU',
    # FP64 counters
    'fp64_add': 'SQ_INSTS_VALU_ADD_F64',
    'fp64_mul': 'SQ_INSTS_VALU_MUL_F64',
    'fp64_fma': 'SQ_INSTS_VALU_FMA_F64',
    'fp64_trans': 'SQ_INSTS_VALU_TRANS_F64',
    'fp64_mfma': 'SQ_INSTS_VALU_MFMA_F64',
    # FP32 counters
    'fp32_add': 'SQ_INSTS_VALU_ADD_F32',
    'fp32_mul': 'SQ_INSTS_VALU_MUL_F32',
    'fp32_fma': 'SQ_INSTS_VALU_FMA_F32',
    'fp32_trans': 'SQ_INSTS_VALU_TRANS_F32',
    # Integer counters
    'int32_total': 'SQ_INSTS_VALU_INT32',
    'int64_total': 'SQ_INSTS_VALU_INT64',
    # VMEM counters
    'vmem_rd': 'SQ_INSTS_VMEM_RD',
    'vmem_wr': 'SQ_INSTS_VMEM_WR',
    # SALU counters
    'salu_total': 'SQ_INSTS_SALU',
    'smem_total': 'SQ_INSTS_SMEM',
    'branch_total': 'SQ_INSTS_BRANCH',
    'flat_total': 'SQ_INSTS_FLAT',
    'lds_total': 'SQ_INSTS_LDS',
    # MFMA
    'mfma_total': 'SQ_INSTS_MFMA',
}

# Reverse mapping for display
ROCPROF_TO_FP64MON = {v: k for k, v in COUNTER_MAP.items()}


def parse_fp64_monitor_csv(filename):
    """
    Parse fp64_monitor CSV and return final cumulative counter values.
    
    fp64_monitor outputs cumulative counters, so we take the max value
    (which is the final sample).
    """
    counters = defaultdict(float)
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            break
        
        # Re-read with csv module
        f.seek(0)
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    
    if not lines:
        return counters
    
    reader = csv.DictReader(lines)
    for row in reader:
        for col, rocprof_name in COUNTER_MAP.items():
            if col in row:
                val = float(row[col])
                if val > counters[rocprof_name]:
                    counters[rocprof_name] = val
    
    return counters


def parse_rocprofv3_csv(filename):
    """
    Parse rocprofv3 counter_collection.csv and sum all per-dispatch values.
    
    rocprofv3 outputs per-kernel-dispatch values, so we sum them all.
    """
    counters = defaultdict(float)
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            counter_name = row.get('Counter_Name', '')
            counter_value = float(row.get('Counter_Value', 0))
            counters[counter_name] += counter_value
    
    return counters


def find_files(patterns):
    """Expand glob patterns and return list of files."""
    files = []
    for pattern in patterns:
        if '*' in pattern or '?' in pattern:
            files.extend(glob.glob(pattern))
        else:
            files.append(pattern)
    return files


def main():
    if len(sys.argv) < 3:
        print("Usage: validate_counters.py <fp64_monitor_csv> <rocprofv3_csv>")
        print()
        print("Examples:")
        print("  validate_counters.py fp64_monitor.csv counter_collection.csv")
        print("  validate_counters.py 'fp64_monitor_output/*.csv' 'rocprof_output/*/*.csv'")
        sys.exit(1)
    
    # Parse arguments - first arg(s) are fp64_monitor, last arg(s) are rocprofv3
    # For simplicity, assume first arg is fp64_monitor pattern, second is rocprofv3 pattern
    fp64_mon_files = find_files([sys.argv[1]])
    rocprof_files = find_files([sys.argv[2]])
    
    if not fp64_mon_files:
        print(f"Error: No fp64_monitor files found matching: {sys.argv[1]}")
        sys.exit(1)
    
    if not rocprof_files:
        print(f"Error: No rocprofv3 files found matching: {sys.argv[2]}")
        sys.exit(1)
    
    # Parse fp64_monitor files (aggregate across multiple files)
    fp64_counters = defaultdict(float)
    for f in fp64_mon_files:
        file_counters = parse_fp64_monitor_csv(f)
        for k, v in file_counters.items():
            fp64_counters[k] += v
    
    # Parse rocprofv3 files (aggregate across multiple files)
    rocprof_counters = defaultdict(float)
    for f in rocprof_files:
        if 'counter_collection' in f:
            file_counters = parse_rocprofv3_csv(f)
            for k, v in file_counters.items():
                rocprof_counters[k] += v
    
    # Print report
    print("=" * 78)
    print("Counter Validation Report")
    print("=" * 78)
    print()
    print(f"fp64_monitor files: {len(fp64_mon_files)}")
    for f in fp64_mon_files:
        print(f"  - {Path(f).name}")
    print()
    print(f"rocprofv3 files: {len([f for f in rocprof_files if 'counter_collection' in f])}")
    for f in rocprof_files:
        if 'counter_collection' in f:
            print(f"  - {Path(f).name}")
    print()
    
    print("-" * 78)
    print(f"{'Counter':<30} {'fp64_monitor':>18} {'rocprofv3':>18} {'Diff %':>10}")
    print("-" * 78)
    
    all_pass = True
    tolerance = 0.05  # 5% tolerance
    
    # Compare each counter
    for rocprof_name in sorted(set(list(fp64_counters.keys()) + list(rocprof_counters.keys()))):
        fp64_val = fp64_counters.get(rocprof_name, 0)
        rocprof_val = rocprof_counters.get(rocprof_name, 0)
        
        if fp64_val == 0 and rocprof_val == 0:
            diff_pct = 0
            status = "SKIP"
        elif rocprof_val == 0:
            diff_pct = 100 if fp64_val > 0 else 0
            status = "FAIL" if fp64_val > 0 else "OK"
        else:
            diff_pct = abs(fp64_val - rocprof_val) / rocprof_val * 100
            status = "OK" if diff_pct <= tolerance * 100 else "FAIL"
        
        if status == "FAIL":
            all_pass = False
        
        # Format output
        if status == "SKIP":
            print(f"{rocprof_name:<30} {'0':>18} {'0':>18} {'(skip)':>10}")
        else:
            print(f"{rocprof_name:<30} {fp64_val:>18,.0f} {rocprof_val:>18,.0f} {diff_pct:>9.2f}%")
    
    print("-" * 78)
    print()
    
    # Summary
    if all_pass:
        print("RESULT: PASS - All counters match within tolerance")
    else:
        print("RESULT: FAIL - Some counters have significant differences")
        print()
        print("Note: Small differences may occur due to:")
        print("  - Sampling timing (fp64_monitor may miss some activity)")
        print("  - Warmup iterations not captured by fp64_monitor")
        print("  - Different GPU agent selection")
    
    print()
    print("=" * 78)
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
