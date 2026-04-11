#!/usr/bin/env python3
"""
Analyze FP64 Monitor CSV output files.

Usage:
    python3 analyze_results.py <csv_file> [csv_file2 ...]
    python3 analyze_results.py fp64_monitor_*.csv
    
When multiple files are provided (e.g., from MPI runs), data is aggregated
across all ranks and a single combined summary is reported.

Output:
    - Console: Aggregated summary
    - fp64_report.txt: Detailed report with per-rank statistics

Supports both:
    - fp64_monitor_tool CSV output (sample_id,elapsed_seconds,...)
    - rocprofv3 counter collection CSV output (Counter_Name,Counter_Value,...)
"""

import sys
import csv
import glob
import statistics
from pathlib import Path
from collections import defaultdict
from io import StringIO


class DualWriter:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, file_handle):
        self.file = file_handle
        self.stdout = sys.stdout
    
    def write(self, text):
        self.stdout.write(text)
        if self.file:
            self.file.write(text)
    
    def flush(self):
        self.stdout.flush()
        if self.file:
            self.file.flush()


def load_fp64_monitor_csv(filename):
    """Load CSV data from a single fp64_monitor output file."""
    samples = []
    summary_line = None
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                summary_line = line.strip()
                continue
            if not line.strip():
                continue
            samples.append(line)
    
    if not samples:
        return [], summary_line
    
    reader = csv.DictReader(samples)
    rows = list(reader)
    return rows, summary_line


def compute_stats(values):
    """Compute statistical measures for a list of values."""
    if not values:
        return None
    n = len(values)
    if n == 1:
        return {
            'min': values[0], 'max': values[0], 'mean': values[0],
            'median': values[0], 'stdev': 0, 'variance': 0
        }
    return {
        'min': min(values),
        'max': max(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if n > 1 else 0,
        'variance': statistics.variance(values) if n > 1 else 0
    }


def write_detailed_report(filenames, report_file=None):
    """Write detailed report with per-rank statistics to file."""
    
    # Load data per rank
    rank_data = {}
    for filename in filenames:
        rows, _ = load_fp64_monitor_csv(filename)
        if rows:
            rank_data[filename] = rows
    
    if not rank_data:
        return None
    
    # Determine report file location (same directory as input files)
    if report_file is None:
        input_dir = str(Path(filenames[0]).parent)
        if input_dir and input_dir != '.':
            report_file = f"{input_dir}/fp64_report.txt"
        else:
            report_file = "fp64_report.txt"
    
    num_ranks = len(rank_data)
    
    with open(report_file, 'w') as f:
        f.write("=" * 78 + "\n")
        f.write("FP64 Monitor Detailed Report\n")
        f.write("=" * 78 + "\n\n")
        
        f.write(f"Input files: {num_ranks}\n")
        for fn in sorted(filenames):
            f.write(f"  - {fn}\n")
        f.write("\n")
        
        # Compute per-rank totals
        # Note: Counters are cumulative, so use max() to get final value per rank
        rank_totals = {}
        for filename, rows in rank_data.items():
            active = [r for r in rows if float(r.get('total_instructions', 0)) > 0]
            if active:
                rank_totals[filename] = {
                    'fp64_add': max(float(r.get('fp64_add', 0)) for r in active),
                    'fp64_mul': max(float(r.get('fp64_mul', 0)) for r in active),
                    'fp64_fma': max(float(r.get('fp64_fma', 0)) for r in active),
                    'fp64_trans': max(float(r.get('fp64_trans', 0)) for r in active),
                    'fp64_mfma': max(float(r.get('fp64_mfma', 0)) for r in active),
                    'total_valu': max(float(r.get('valu_instructions', 0)) for r in active),
                    'total_insts': max(float(r.get('total_instructions', 0)) for r in active),
                    'gpu_hwm': max((float(r.get('gpu_mem_hwm_mb', 0)) for r in rows), default=0),
                    'vram_hwm': max((float(r.get('vram_hwm_mb', 0)) for r in rows), default=0),
                    'gtt_hwm': max((float(r.get('gtt_hwm_mb', 0)) for r in rows), default=0),
                    'cpu_hwm': max((float(r.get('cpu_resident_hwm_mb', 0)) for r in rows), default=0),
                    'active_samples': len(active),
                }
                # Include MFMA since SQ_INSTS_VALU includes MFMA instructions
                rank_totals[filename]['fp64_valu'] = (
                    rank_totals[filename]['fp64_add'] + 
                    rank_totals[filename]['fp64_mul'] + 
                    rank_totals[filename]['fp64_fma'] + 
                    rank_totals[filename]['fp64_trans'] +
                    rank_totals[filename]['fp64_mfma']
                )
                # Calculate percentage from computed totals
                if rank_totals[filename]['total_valu'] > 0:
                    rank_totals[filename]['fp64_valu_pct'] = (
                        rank_totals[filename]['fp64_valu'] / rank_totals[filename]['total_valu'] * 100
                    )
                else:
                    rank_totals[filename]['fp64_valu_pct'] = 0.0
        
        if not rank_totals:
            f.write("No active samples found in any rank.\n")
            return
        
        # Per-rank summary table
        f.write("-" * 78 + "\n")
        f.write("Per-Rank Summary\n")
        f.write("-" * 78 + "\n\n")
        
        f.write(f"{'Rank':<6} {'Samples':>8} {'FP64 VALU':>15} {'Total VALU':>15} {'FP64%':>8} {'GPU HWM':>10}\n")
        f.write("-" * 78 + "\n")
        
        for i, filename in enumerate(sorted(rank_totals.keys())):
            data = rank_totals[filename]
            f.write(f"{i:<6} {data['active_samples']:>8} {data['fp64_valu']:>15,.0f} "
                   f"{data['total_valu']:>15,.0f} {data['fp64_valu_pct']:>7.2f}% "
                   f"{data['gpu_hwm']:>9.1f}M\n")
        f.write("\n")
        
        # Statistical analysis across ranks
        f.write("-" * 78 + "\n")
        f.write("Statistical Analysis Across MPI Ranks\n")
        f.write("-" * 78 + "\n\n")
        
        metrics = [
            ('FP64 VALU Instructions', 'fp64_valu'),
            ('Total VALU Instructions', 'total_valu'),
            ('Total Instructions', 'total_insts'),
            ('FP64 % of VALU', 'fp64_valu_pct'),
            ('GPU Memory HWM (MB)', 'gpu_hwm'),
            ('VRAM HWM (MB)', 'vram_hwm'),
            ('GTT HWM (MB)', 'gtt_hwm'),
            ('CPU Memory HWM (MB)', 'cpu_hwm'),
        ]
        
        for label, key in metrics:
            values = [rank_totals[fn][key] for fn in rank_totals]
            stats = compute_stats(values)
            if stats and stats['max'] > 0:  # Skip metrics with all zeros
                f.write(f"{label}:\n")
                if key in ['fp64_valu_pct']:
                    f.write(f"  Min:      {stats['min']:>12.4f}%\n")
                    f.write(f"  Max:      {stats['max']:>12.4f}%\n")
                    f.write(f"  Mean:     {stats['mean']:>12.4f}%\n")
                    f.write(f"  Median:   {stats['median']:>12.4f}%\n")
                    f.write(f"  Std Dev:  {stats['stdev']:>12.4f}\n")
                elif key in ['gpu_hwm', 'vram_hwm', 'gtt_hwm', 'cpu_hwm']:
                    f.write(f"  Min:      {stats['min']:>12.1f}\n")
                    f.write(f"  Max:      {stats['max']:>12.1f}\n")
                    f.write(f"  Mean:     {stats['mean']:>12.1f}\n")
                    f.write(f"  Median:   {stats['median']:>12.1f}\n")
                    f.write(f"  Std Dev:  {stats['stdev']:>12.2f}\n")
                else:
                    f.write(f"  Min:      {stats['min']:>18,.0f}\n")
                    f.write(f"  Max:      {stats['max']:>18,.0f}\n")
                    f.write(f"  Mean:     {stats['mean']:>18,.0f}\n")
                    f.write(f"  Median:   {stats['median']:>18,.0f}\n")
                    f.write(f"  Std Dev:  {stats['stdev']:>18,.0f}\n")
                f.write("\n")
        
        # Load imbalance analysis
        fp64_values = [rank_totals[fn]['fp64_valu'] for fn in rank_totals]
        if fp64_values and max(fp64_values) > 0:
            imbalance = (max(fp64_values) - min(fp64_values)) / max(fp64_values) * 100
            f.write("-" * 78 + "\n")
            f.write("Load Balance Analysis\n")
            f.write("-" * 78 + "\n\n")
            f.write(f"FP64 VALU Load Imbalance: {imbalance:.2f}%\n")
            f.write(f"  (0% = perfectly balanced, 100% = completely imbalanced)\n\n")
        
        f.write("=" * 78 + "\n")
    
    return report_file


def write_detailed_report_to_file(filenames, f):
    """Write detailed per-rank statistics to an already open file handle."""
    
    # Load data per rank
    rank_data = {}
    for filename in filenames:
        rows, _ = load_fp64_monitor_csv(filename)
        if rows:
            rank_data[filename] = rows
    
    if not rank_data:
        return
    
    num_ranks = len(rank_data)
    
    f.write("=" * 78 + "\n")
    f.write("Detailed Per-Rank Statistics\n")
    f.write("=" * 78 + "\n\n")
    
    f.write(f"Input files: {num_ranks}\n")
    for fn in sorted(filenames):
        f.write(f"  - {fn}\n")
    f.write("\n")
    
    # Compute per-rank totals
    # Note: Counters are cumulative, so use max() to get final value per rank
    rank_totals = {}
    for filename, rows in rank_data.items():
        active = [r for r in rows if float(r.get('total_instructions', 0)) > 0]
        if active:
            rank_totals[filename] = {
                'fp64_add': max(float(r.get('fp64_add', 0)) for r in active),
                'fp64_mul': max(float(r.get('fp64_mul', 0)) for r in active),
                'fp64_fma': max(float(r.get('fp64_fma', 0)) for r in active),
                'fp64_trans': max(float(r.get('fp64_trans', 0)) for r in active),
                'fp64_mfma': max(float(r.get('fp64_mfma', 0)) for r in active),
                'total_valu': max(float(r.get('valu_instructions', 0)) for r in active),
                'total_insts': max(float(r.get('total_instructions', 0)) for r in active),
                'gpu_hwm': max((float(r.get('gpu_mem_hwm_mb', 0)) for r in rows), default=0),
                'vram_hwm': max((float(r.get('vram_hwm_mb', 0)) for r in rows), default=0),
                'gtt_hwm': max((float(r.get('gtt_hwm_mb', 0)) for r in rows), default=0),
                'cpu_hwm': max((float(r.get('cpu_resident_hwm_mb', 0)) for r in rows), default=0),
                'active_samples': len(active),
            }
            # Include MFMA since SQ_INSTS_VALU includes MFMA instructions
            rank_totals[filename]['fp64_valu'] = (
                rank_totals[filename]['fp64_add'] + 
                rank_totals[filename]['fp64_mul'] + 
                rank_totals[filename]['fp64_fma'] + 
                rank_totals[filename]['fp64_trans'] +
                rank_totals[filename]['fp64_mfma']
            )
            # Calculate percentage from computed totals
            if rank_totals[filename]['total_valu'] > 0:
                rank_totals[filename]['fp64_valu_pct'] = (
                    rank_totals[filename]['fp64_valu'] / rank_totals[filename]['total_valu'] * 100
                )
            else:
                rank_totals[filename]['fp64_valu_pct'] = 0.0
    
    if not rank_totals:
        f.write("No active samples found in any rank.\n")
        return
    
    # Per-rank summary table
    f.write("-" * 78 + "\n")
    f.write("Per-Rank Summary\n")
    f.write("-" * 78 + "\n\n")
    
    f.write(f"{'Rank':<6} {'Samples':>8} {'FP64 VALU':>15} {'Total VALU':>15} {'FP64%':>8} {'GPU HWM':>10}\n")
    f.write("-" * 78 + "\n")
    
    for i, filename in enumerate(sorted(rank_totals.keys())):
        data = rank_totals[filename]
        f.write(f"{i:<6} {data['active_samples']:>8} {data['fp64_valu']:>15,.0f} "
               f"{data['total_valu']:>15,.0f} {data['fp64_valu_pct']:>7.2f}% "
               f"{data['gpu_hwm']:>9.1f}M\n")
    f.write("\n")
    
    # Statistical analysis across ranks
    f.write("-" * 78 + "\n")
    f.write("Statistical Analysis Across MPI Ranks\n")
    f.write("-" * 78 + "\n\n")
    
    metrics = [
        ('FP64 VALU Instructions', 'fp64_valu'),
        ('Total VALU Instructions', 'total_valu'),
        ('Total Instructions', 'total_insts'),
        ('FP64 % of VALU', 'fp64_valu_pct'),
        ('GPU Memory HWM (MB)', 'gpu_hwm'),
        ('VRAM HWM (MB)', 'vram_hwm'),
        ('GTT HWM (MB)', 'gtt_hwm'),
        ('CPU Memory HWM (MB)', 'cpu_hwm'),
    ]
    
    for label, key in metrics:
        values = [rank_totals[fn][key] for fn in rank_totals]
        stats = compute_stats(values)
        if stats and stats['max'] > 0:  # Skip metrics with all zeros
            f.write(f"{label}:\n")
            if key in ['fp64_valu_pct']:
                f.write(f"  Min:      {stats['min']:>12.4f}%\n")
                f.write(f"  Max:      {stats['max']:>12.4f}%\n")
                f.write(f"  Mean:     {stats['mean']:>12.4f}%\n")
                f.write(f"  Median:   {stats['median']:>12.4f}%\n")
                f.write(f"  Std Dev:  {stats['stdev']:>12.4f}\n")
            elif key in ['gpu_hwm', 'vram_hwm', 'gtt_hwm', 'cpu_hwm']:
                f.write(f"  Min:      {stats['min']:>12.1f}\n")
                f.write(f"  Max:      {stats['max']:>12.1f}\n")
                f.write(f"  Mean:     {stats['mean']:>12.1f}\n")
                f.write(f"  Median:   {stats['median']:>12.1f}\n")
                f.write(f"  Std Dev:  {stats['stdev']:>12.2f}\n")
            else:
                f.write(f"  Min:      {stats['min']:>18,.0f}\n")
                f.write(f"  Max:      {stats['max']:>18,.0f}\n")
                f.write(f"  Mean:     {stats['mean']:>18,.0f}\n")
                f.write(f"  Median:   {stats['median']:>18,.0f}\n")
                f.write(f"  Std Dev:  {stats['stdev']:>18,.0f}\n")
            f.write("\n")
    
    # Load imbalance analysis
    fp64_values = [rank_totals[fn]['fp64_valu'] for fn in rank_totals]
    if fp64_values and max(fp64_values) > 0:
        imbalance = (max(fp64_values) - min(fp64_values)) / max(fp64_values) * 100
        f.write("-" * 78 + "\n")
        f.write("Load Balance Analysis\n")
        f.write("-" * 78 + "\n\n")
        f.write(f"FP64 VALU Load Imbalance: {imbalance:.2f}%\n")
        f.write(f"  (0% = perfectly balanced, 100% = completely imbalanced)\n\n")
    
    f.write("=" * 78 + "\n")


def analyze_fp64_monitor_files(filenames):
    """Analyze and aggregate CSV output from one or more fp64_monitor files."""
    
    all_rows = []
    num_ranks = len(filenames)
    
    for filename in filenames:
        rows, _ = load_fp64_monitor_csv(filename)
        for row in rows:
            row['_source_file'] = filename
        all_rows.extend(rows)
    
    if not all_rows:
        print("Error: No data found in CSV files")
        return
    
    # Determine report file location (same directory as input files)
    input_dir = str(Path(filenames[0]).parent)
    if input_dir and input_dir != '.':
        report_file = f"{input_dir}/fp64_report.txt"
    else:
        report_file = "fp64_report.txt"
    
    # Open report file and create dual writer for stdout + file
    with open(report_file, 'w') as f:
        out = DualWriter(f)
        
        # Helper to print to both stdout and file
        def p(text=""):
            print(text, file=out)
        
        # Write summary (to both stdout and file)
        p("=" * 70)
        p("FP64 Monitor Analysis Summary")
        p("=" * 70)
        p()
        p(f"MPI Ranks:     {num_ranks}")
        p(f"Total samples: {len(all_rows)}")
        
        # Find samples with actual activity (non-zero instructions)
        active_samples = [r for r in all_rows if float(r.get('total_instructions', 0)) > 0]
        idle_samples = [r for r in all_rows if float(r.get('total_instructions', 0)) == 0]
        
        p(f"Active samples: {len(active_samples)}")
        p(f"Idle samples:   {len(idle_samples)}")
        p()
        
        if not active_samples:
            p("WARNING: No samples with GPU activity detected!")
            p()
        
        if active_samples:
            # Group active samples by rank (source file) for proper aggregation
            # Counters are cumulative, so we take max per rank, then sum across ranks
            rank_samples = defaultdict(list)
            for r in active_samples:
                rank_samples[r.get('_source_file', 'unknown')].append(r)
            
            def sum_across_ranks(counter_name):
                """Sum the final (max) value of a counter across all ranks."""
                total = 0
                for samples in rank_samples.values():
                    rank_max = max((float(r.get(counter_name, 0)) for r in samples), default=0)
                    total += rank_max
                return total
            
            # Get totals that are common across all counter sets
            total_valu = sum_across_ranks('valu_instructions')
            total_insts = sum_across_ranks('total_instructions')
            
            # Show general statistics if we have totals
            if total_valu > 0 or total_insts > 0:
                p("-" * 70)
                p("General Statistics")
                p("-" * 70)
                p()
                p(f"  Total VALU:      {total_valu:>18,.0f}")
                p(f"  Total Instr:     {total_insts:>18,.0f}")
                p()
            
            # FP64 Instruction Counts (only if data present)
            total_fp64_add = sum_across_ranks('fp64_add')
            total_fp64_mul = sum_across_ranks('fp64_mul')
            total_fp64_fma = sum_across_ranks('fp64_fma')
            total_fp64_trans = sum_across_ranks('fp64_trans')
            total_fp64_mfma = sum_across_ranks('fp64_mfma')
            # Include MFMA since SQ_INSTS_VALU includes MFMA instructions
            total_fp64_valu = total_fp64_add + total_fp64_mul + total_fp64_fma + total_fp64_trans + total_fp64_mfma
            
            if total_fp64_valu > 0:
                p("-" * 70)
                p("FP64 Instruction Counts (aggregated across all ranks)")
                p("-" * 70)
                p()
                p(f"  FP64 ADD:        {total_fp64_add:>18,.0f}")
                p(f"  FP64 MUL:        {total_fp64_mul:>18,.0f}")
                p(f"  FP64 FMA:        {total_fp64_fma:>18,.0f}")
                p(f"  FP64 Trans:      {total_fp64_trans:>18,.0f}")
                p(f"  FP64 MFMA:       {total_fp64_mfma:>18,.0f}")
                p(f"  ----------------------------------------")
                p(f"  Total FP64 VALU: {total_fp64_valu:>18,.0f}")
                p()
                
                # Percentages
                if total_valu > 0:
                    valu_pct = (total_fp64_valu / total_valu) * 100
                    p(f"FP64 %% of VALU:   {valu_pct:>10.2f}%")
                
                if total_insts > 0:
                    total_pct = (total_fp64_valu / total_insts) * 100
                    p(f"FP64 %% of Total:  {total_pct:>10.2f}%")
                p()
            
            # FP32 Instruction Counts (only if data present)
            total_fp32_add = sum_across_ranks('fp32_add')
            total_fp32_mul = sum_across_ranks('fp32_mul')
            total_fp32_fma = sum_across_ranks('fp32_fma')
            total_fp32_trans = sum_across_ranks('fp32_trans')
            total_fp32_mfma = sum_across_ranks('fp32_mfma')
            total_fp32_valu = total_fp32_add + total_fp32_mul + total_fp32_fma + total_fp32_trans + total_fp32_mfma
            
            if total_fp32_valu > 0:
                p("-" * 70)
                p("FP32 Instruction Counts (aggregated across all ranks)")
                p("-" * 70)
                p()
                p(f"  FP32 ADD:        {total_fp32_add:>18,.0f}")
                p(f"  FP32 MUL:        {total_fp32_mul:>18,.0f}")
                p(f"  FP32 FMA:        {total_fp32_fma:>18,.0f}")
                p(f"  FP32 Trans:      {total_fp32_trans:>18,.0f}")
                p(f"  FP32 MFMA:       {total_fp32_mfma:>18,.0f}")
                p(f"  ----------------------------------------")
                p(f"  Total FP32 VALU: {total_fp32_valu:>18,.0f}")
                p()
                if total_valu > 0:
                    p(f"FP32 %% of VALU:   {(total_fp32_valu / total_valu) * 100:>10.2f}%")
                if total_insts > 0:
                    p(f"FP32 %% of Total:  {(total_fp32_valu / total_insts) * 100:>10.2f}%")
                p()
            
            # FP16/BF16 Instruction Counts (only if data present)
            total_fp16_add = sum_across_ranks('fp16_add')
            total_fp16_mul = sum_across_ranks('fp16_mul')
            total_fp16_fma = sum_across_ranks('fp16_fma')
            total_fp16_mfma = sum_across_ranks('fp16_mfma')
            total_bf16_mfma = sum_across_ranks('bf16_mfma')
            total_fp16_valu = total_fp16_add + total_fp16_mul + total_fp16_fma + total_fp16_mfma + total_bf16_mfma
            
            if total_fp16_valu > 0:
                p("-" * 70)
                p("FP16/BF16 Instruction Counts (aggregated across all ranks)")
                p("-" * 70)
                p()
                p(f"  FP16 ADD:        {total_fp16_add:>18,.0f}")
                p(f"  FP16 MUL:        {total_fp16_mul:>18,.0f}")
                p(f"  FP16 FMA:        {total_fp16_fma:>18,.0f}")
                p(f"  FP16 MFMA:       {total_fp16_mfma:>18,.0f}")
                p(f"  BF16 MFMA:       {total_bf16_mfma:>18,.0f}")
                p(f"  ----------------------------------------")
                p(f"  Total FP16/BF16: {total_fp16_valu:>18,.0f}")
                p()
                if total_valu > 0:
                    p(f"FP16/BF16 %% of VALU:  {(total_fp16_valu / total_valu) * 100:>10.2f}%")
                if total_insts > 0:
                    p(f"FP16/BF16 %% of Total: {(total_fp16_valu / total_insts) * 100:>10.2f}%")
                p()
            
            # Integer Instruction Counts (only if data present)
            total_int32 = sum_across_ranks('int32_total')
            total_int64 = sum_across_ranks('int64_total')
            total_int_valu = total_int32 + total_int64
            
            if total_int_valu > 0:
                p("-" * 70)
                p("Integer Instruction Counts (aggregated across all ranks)")
                p("-" * 70)
                p()
                p(f"  INT32 VALU:      {total_int32:>18,.0f}")
                p(f"  INT64 VALU:      {total_int64:>18,.0f}")
                p(f"  ----------------------------------------")
                p(f"  Total INT VALU:  {total_int_valu:>18,.0f}")
                p()
                if total_valu > 0:
                    p(f"INT %% of VALU:    {(total_int_valu / total_valu) * 100:>10.2f}%")
                if total_insts > 0:
                    p(f"INT %% of Total:   {(total_int_valu / total_insts) * 100:>10.2f}%")
                p()
            
            # Vector Memory (VMEM) Analysis (only if data present)
            total_vmem_rd = sum_across_ranks('vmem_rd')
            total_vmem_wr = sum_across_ranks('vmem_wr')
            total_vmem_rd_cycles = sum_across_ranks('vmem_rd_cycles')
            total_vmem_wr_cycles = sum_across_ranks('vmem_wr_cycles')
            total_vmem = total_vmem_rd + total_vmem_wr
            
            if total_vmem > 0:
                p("-" * 70)
                p("Vector Memory (VMEM) Instructions (aggregated across all ranks)")
                p("-" * 70)
                p()
                p(f"  VMEM Reads:      {total_vmem_rd:>18,.0f}")
                p(f"  VMEM Writes:     {total_vmem_wr:>18,.0f}")
                p(f"  ----------------------------------------")
                p(f"  Total VMEM:      {total_vmem:>18,.0f}")
                p()
                if total_insts > 0:
                    p(f"VMEM Read %% of Total:   {(total_vmem_rd / total_insts) * 100:>10.2f}%")
                    p(f"VMEM Write %% of Total:  {(total_vmem_wr / total_insts) * 100:>10.2f}%")
                    p(f"VMEM Total %% of Total:  {(total_vmem / total_insts) * 100:>10.2f}%")
                if total_vmem_wr > 0:
                    p(f"Read/Write Ratio:        {total_vmem_rd / total_vmem_wr:>10.2f}")
                p()
                
                # Cycle/latency analysis (only if cycle data present)
                # Note: These are ISSUE cycles (time to send addr/cmd), not full memory latency
                if total_vmem_rd_cycles > 0 or total_vmem_wr_cycles > 0:
                    p(f"  VMEM Read Issue Cycles:  {total_vmem_rd_cycles:>14,.0f}")
                    p(f"  VMEM Write Issue Cycles: {total_vmem_wr_cycles:>14,.0f}")
                    p()
                    if total_vmem_rd > 0 and total_vmem_rd_cycles > 0:
                        avg_rd_latency = total_vmem_rd_cycles / total_vmem_rd
                        p(f"Avg Read Issue Time:    {avg_rd_latency:>10.1f} cycles/instr")
                    if total_vmem_wr > 0 and total_vmem_wr_cycles > 0:
                        avg_wr_latency = total_vmem_wr_cycles / total_vmem_wr
                        p(f"Avg Write Issue Time:   {avg_wr_latency:>10.1f} cycles/instr")
                    p(f"  (Issue time = cycles to send addr/cmd, not full memory latency)")
                    p()
            
            # MFMA / GEMM Analysis
            total_mfma = sum_across_ranks('mfma_total')
            
            # Get INT8 MFMA from CSV if available (FP32/FP16/BF16 already computed above)
            total_int8_mfma = sum_across_ranks('int8_mfma')
            
            if total_mfma > 0 or total_fp64_mfma > 0:
                p("-" * 70)
                p("MFMA / GEMM Analysis")
                p("-" * 70)
                p()
                
                # GEMM likelihood based on MFMA percentage of total instructions
                # Using total_insts gives a better picture than VALU since VALU includes overhead
                mfma_pct = (total_mfma / total_insts * 100) if total_insts > 0 else 0
                
                p(f"  Total MFMA:      {total_mfma:>18,.0f}")
                if total_fp64_mfma > 0:
                    p(f"    FP64 MFMA:     {total_fp64_mfma:>18,.0f}")
                if total_fp32_mfma > 0:
                    p(f"    FP32 MFMA:     {total_fp32_mfma:>18,.0f}")
                if total_fp16_mfma > 0:
                    p(f"    FP16 MFMA:     {total_fp16_mfma:>18,.0f}")
                if total_bf16_mfma > 0:
                    p(f"    BF16 MFMA:     {total_bf16_mfma:>18,.0f}")
                if total_int8_mfma > 0:
                    p(f"    INT8 MFMA:     {total_int8_mfma:>18,.0f}")
                p()
                
                p(f"MFMA %% of Total:   {mfma_pct:>9.2f}%")
                
                # GEMM likelihood heuristic based on MFMA % of total instructions
                # Even efficient GEMM typically shows 5-10% MFMA due to overhead
                if mfma_pct >= 10:
                    gemm_likelihood = "Very High"
                    gemm_desc = "Workload is dominated by matrix operations (GEMM/convolution)"
                elif mfma_pct >= 3:
                    gemm_likelihood = "High"
                    gemm_desc = "Significant matrix operation content"
                elif mfma_pct >= 1:
                    gemm_likelihood = "Moderate"
                    gemm_desc = "Some matrix operations present"
                elif mfma_pct > 0:
                    gemm_likelihood = "Low"
                    gemm_desc = "Minimal matrix operation content"
                else:
                    gemm_likelihood = "None"
                    gemm_desc = "No MFMA instructions detected"
                
                p(f"GEMM Likelihood:   {gemm_likelihood}")
                p(f"  ({gemm_desc})")
                p()
                
                # Estimate TFLOPS from MFMA counts
                # FLOPs per MFMA instruction (approximate, depends on specific instruction variant):
                #   FP64 MFMA: 128 FLOPs (4x4x4 matrix = 64 FMA = 128 FLOPs)
                #   FP32 MFMA: 512 FLOPs (16x16x4 matrix = 256 FMA = 512 FLOPs)
                #   FP16/BF16 MFMA: 1024 FLOPs (16x16x16 or 32x32x8)
                #   INT8 MFMA: 2048 OPs (32x32x16)
                
                # Get elapsed time for rate calculation
                elapsed_times = [float(r.get('elapsed_seconds', 0)) for r in active_samples]
                if elapsed_times:
                    max_elapsed = max(elapsed_times)
                    min_elapsed = min(elapsed_times)
                    duration = max_elapsed - min_elapsed if len(elapsed_times) > 1 else max_elapsed
                    
                    if duration > 0:
                        # Calculate estimated FLOPs
                        est_fp64_flops = total_fp64_mfma * 128
                        est_fp32_flops = total_fp32_mfma * 512
                        est_fp16_flops = (total_fp16_mfma + total_bf16_mfma) * 1024
                        est_int8_ops = total_int8_mfma * 2048
                        
                        total_est_flops = est_fp64_flops + est_fp32_flops + est_fp16_flops
                        
                        p("Estimated MFMA Performance:")
                        p(f"  Duration:        {duration:>10.3f} seconds")
                        
                        if est_fp64_flops > 0:
                            tflops_fp64 = est_fp64_flops / duration / 1e12
                            p(f"  FP64 MFMA:       {tflops_fp64:>10.3f} TFLOPS")
                        if est_fp32_flops > 0:
                            tflops_fp32 = est_fp32_flops / duration / 1e12
                            p(f"  FP32 MFMA:       {tflops_fp32:>10.3f} TFLOPS")
                        if est_fp16_flops > 0:
                            tflops_fp16 = est_fp16_flops / duration / 1e12
                            p(f"  FP16/BF16 MFMA:  {tflops_fp16:>10.3f} TFLOPS")
                        if est_int8_ops > 0:
                            tops_int8 = est_int8_ops / duration / 1e12
                            p(f"  INT8 MFMA:       {tops_int8:>10.3f} TOPS")
                        
                        if total_est_flops > 0:
                            total_tflops = total_est_flops / duration / 1e12
                            p(f"  ----------------------------------------")
                            p(f"  Total MFMA:      {total_tflops:>10.3f} TFLOPS")
                        
                        p()
                        p("  Note: TFLOPS estimates assume common MFMA instruction variants.")
                        p("        Actual performance may vary based on specific instructions used.")
                p()
        
        # Memory Analysis - compute per-rank HWMs then aggregate
        has_memory_data = 'gpu_mem_used_mb' in all_rows[0] if all_rows else False
        if has_memory_data:
            p("-" * 70)
            p("Memory Usage (aggregated across all ranks)")
            p("-" * 70)
            p()
            
            # Group rows by source file (rank) and compute per-rank HWMs
            rank_rows = defaultdict(list)
            for row in all_rows:
                rank_rows[row.get('_source_file', 'unknown')].append(row)
            
            total_ranks = len(rank_rows)
            
            # Compute per-rank memory HWMs
            per_rank_gpu_hwm = []
            per_rank_vram_hwm = []
            per_rank_gtt_hwm = []
            per_rank_cpu_hwm = []
            gpu_total_per_rank = 0
            vram_total_per_rank = 0
            
            for filename, rows in rank_rows.items():
                gpu_hwm = max((float(r.get('gpu_mem_hwm_mb', 0)) for r in rows), default=0)
                vram_hwm = max((float(r.get('vram_hwm_mb', 0)) for r in rows), default=0)
                gtt_hwm = max((float(r.get('gtt_hwm_mb', 0)) for r in rows), default=0)
                cpu_hwm = max((float(r.get('cpu_resident_hwm_mb', 0)) for r in rows), default=0)
                
                if gpu_hwm > 0:
                    per_rank_gpu_hwm.append(gpu_hwm)
                if vram_hwm > 0:
                    per_rank_vram_hwm.append(vram_hwm)
                if gtt_hwm > 0:
                    per_rank_gtt_hwm.append(gtt_hwm)
                if cpu_hwm > 0:
                    per_rank_cpu_hwm.append(cpu_hwm)
                
                # Get total memory (should be same for all samples in a rank)
                gpu_totals = [float(r.get('gpu_mem_total_mb', 0)) for r in rows if float(r.get('gpu_mem_total_mb', 0)) > 0]
                vram_totals = [float(r.get('vram_total_mb', 0)) for r in rows if float(r.get('vram_total_mb', 0)) > 0]
                if gpu_totals:
                    gpu_total_per_rank = gpu_totals[0]
                if vram_totals:
                    vram_total_per_rank = vram_totals[0]
            
            def rank_count_str(valid_ranks, total_ranks):
                """Format rank count, noting if some ranks had no data."""
                if valid_ranks == total_ranks:
                    return f"{valid_ranks} ranks"
                else:
                    return f"{valid_ranks} of {total_ranks} ranks"
            
            if per_rank_gpu_hwm:
                stats = compute_stats(per_rank_gpu_hwm)
                total_hwm = sum(per_rank_gpu_hwm)
                p(f"GPU Memory (HIP): {gpu_total_per_rank/1024:.1f} GB per rank")
                p(f"  HWM Min:    {stats['min']:>10.1f} MB")
                p(f"  HWM Max:    {stats['max']:>10.1f} MB")
                p(f"  HWM Mean:   {stats['mean']:>10.1f} MB")
                p(f"  HWM StdDev: {stats['stdev']:>10.2f} MB")
                p(f"  HWM Total:  {total_hwm/1024:>10.2f} GB ({rank_count_str(len(per_rank_gpu_hwm), total_ranks)})")
                p()
            
            if per_rank_vram_hwm:
                stats = compute_stats(per_rank_vram_hwm)
                total_hwm = sum(per_rank_vram_hwm)
                p(f"VRAM (rocm-smi): {vram_total_per_rank/1024:.1f} GB per rank")
                p(f"  HWM Min:    {stats['min']:>10.1f} MB")
                p(f"  HWM Max:    {stats['max']:>10.1f} MB")
                p(f"  HWM Mean:   {stats['mean']:>10.1f} MB")
                p(f"  HWM StdDev: {stats['stdev']:>10.2f} MB")
                p(f"  HWM Total:  {total_hwm/1024:>10.2f} GB ({rank_count_str(len(per_rank_vram_hwm), total_ranks)})")
                p()
            
            if per_rank_gtt_hwm:
                stats = compute_stats(per_rank_gtt_hwm)
                total_hwm = sum(per_rank_gtt_hwm)
                p(f"GTT (system memory for GPU):")
                p(f"  HWM Min:    {stats['min']:>10.1f} MB")
                p(f"  HWM Max:    {stats['max']:>10.1f} MB")
                p(f"  HWM Mean:   {stats['mean']:>10.1f} MB")
                p(f"  HWM StdDev: {stats['stdev']:>10.2f} MB")
                p(f"  HWM Total:  {total_hwm/1024:>10.2f} GB ({rank_count_str(len(per_rank_gtt_hwm), total_ranks)})")
                p()
            
            if per_rank_cpu_hwm:
                stats = compute_stats(per_rank_cpu_hwm)
                total_hwm = sum(per_rank_cpu_hwm)
                p(f"CPU Resident Memory:")
                p(f"  HWM Min:    {stats['min']:>10.1f} MB")
                p(f"  HWM Max:    {stats['max']:>10.1f} MB")
                p(f"  HWM Mean:   {stats['mean']:>10.1f} MB")
                p(f"  HWM StdDev: {stats['stdev']:>10.2f} MB")
                p(f"  HWM Total:  {total_hwm/1024:>10.2f} GB ({rank_count_str(len(per_rank_cpu_hwm), total_ranks)})")
                p()
        
        p("=" * 70)
        
        # Now write the detailed per-rank report (file only, not stdout)
        f.write("\n\n")
        write_detailed_report_to_file(filenames, f)
    
    print(f"\nReport written to: {report_file}")


def analyze_rocprofv3_csv(filename):
    """Analyze CSV output from rocprofv3 (legacy support)."""
    
    counters = defaultdict(list)
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            counter_name = row['Counter_Name']
            counter_value = float(row['Counter_Value'])
            counters[counter_name].append(counter_value)
    
    print("=" * 70)
    print("rocprofv3 Counter Analysis")
    print("=" * 70)
    print()
    
    if not counters:
        print("No counters found in file")
        return
    
    dispatch_count = len(list(counters.values())[0]) if counters else 0
    print(f"Total kernel dispatches: {dispatch_count}")
    print()
    
    print("Summed across all dispatches:")
    for name in sorted(counters.keys()):
        total = sum(counters[name])
        print(f"  {name:30s}: {total:>15,.0f}")
    
    # FP64 analysis
    total_fp64_fma = sum(counters.get('SQ_INSTS_VALU_FMA_F64', [0]))
    total_fp64_add = sum(counters.get('SQ_INSTS_VALU_ADD_F64', [0]))
    total_fp64_mul = sum(counters.get('SQ_INSTS_VALU_MUL_F64', [0]))
    total_valu = sum(counters.get('SQ_INSTS_VALU', [0]))
    
    total_fp64 = total_fp64_fma + total_fp64_add + total_fp64_mul
    
    print()
    print(f"Total FP64 VALU: {total_fp64:,.0f}")
    print(f"Total VALU:      {total_valu:,.0f}")
    if total_valu > 0:
        print(f"FP64 %:          {(total_fp64/total_valu)*100:.2f}%")
    
    print()
    print("=" * 70)


def detect_format(filename):
    """Detect CSV format by examining header."""
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if 'Counter_Name' in line:
                return 'rocprofv3'
            elif 'sample_id' in line:
                return 'fp64_monitor'
            else:
                return 'unknown'
    return 'unknown'


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_results.py <csv_file> [csv_file2 ...]")
        print("       analyze_results.py fp64_monitor_*.csv")
        print()
        print("When multiple files are provided, data is aggregated across all ranks.")
        sys.exit(1)
    
    # Collect all input files (handle glob patterns)
    filenames = []
    for arg in sys.argv[1:]:
        if '*' in arg or '?' in arg:
            filenames.extend(sorted(glob.glob(arg)))
        else:
            filenames.append(arg)
    
    if not filenames:
        print("Error: No files found")
        sys.exit(1)
    
    # Verify all files exist
    for f in filenames:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)
    
    # Detect format from first file
    fmt = detect_format(filenames[0])
    
    if fmt == 'fp64_monitor':
        analyze_fp64_monitor_files(filenames)
    elif fmt == 'rocprofv3':
        if len(filenames) > 1:
            print("Warning: Multiple rocprofv3 files not yet supported, using first file")
        analyze_rocprofv3_csv(filenames[0])
    else:
        print(f"Error: Unknown CSV format in {filenames[0]}")
        print("Expected either fp64_monitor or rocprofv3 format")
        sys.exit(1)


if __name__ == '__main__':
    main()
