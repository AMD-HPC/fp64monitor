#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE in the project root for license information.

"""
Analyze FP64 Monitor CSV output files.

Usage:
    python3 fp64_monitor_analyze_results.py <csv_file> [csv_file2 ...]
    python3 fp64_monitor_analyze_results.py fp64_monitor_*.csv
    
When multiple files are provided, data is aggregated and a single combined
summary is reported. Filenames are classified when possible:
    - Same PID, multiple ``_gpu<N>_pid`` files → one process, multiple GPUs.
    - ``_gpu<N>_rank<R>`` or legacy ``fp64_monitor_<rank>.csv`` → MPI ranks.

Deduplication:
    When multiple processes on the same host monitor the same GPU, hardware
    counters are shared (per-GPU, not per-process). The script automatically
    deduplicates by (hostname, gpu_id), keeping only the lowest MPI rank (or
    first file when rank is unknown) per (hostname, gpu_id).

Output:
    - Console: Aggregated summary
    - fp64_report.txt: Detailed per-source statistics

Supports both:
    - fp64_monitor_tool CSV output (sample_id,elapsed_seconds,...)
    - rocprofv3 counter collection CSV output (Counter_Name,Counter_Value,...)
"""

import sys
import csv
import glob
import re
import statistics
from pathlib import Path
from collections import defaultdict
from io import StringIO


def parse_filename_info(filename):
    """
    Parse hostname, GPU ID, and rank from filename.
    
    Expected formats:
      - fp64_monitor_<hostname>_gpu<gpu_id>_rank<rank>.csv  (MPI mode)
      - fp64_monitor_<hostname>_gpu<gpu_id>_pid<pid>.csv   (non-MPI mode)
      - fp64_monitor_<rank>.csv                            (legacy MPI mode)
      - fp64_monitor_<hostname>_<pid>.csv                  (legacy non-MPI mode)
    
    Returns: dict with 'hostname', 'gpu_id', 'rank' (or None if not found)
    """
    basename = Path(filename).stem  # Remove directory and extension
    
    # Try new format: fp64_monitor_<hostname>_gpu<gpu_id>_rank<rank>
    match = re.match(r'fp64_monitor_(.+)_gpu(\d+)_rank(\d+)$', basename)
    if match:
        return {
            'hostname': match.group(1),
            'gpu_id': int(match.group(2)),
            'rank': int(match.group(3)),
            'filename': filename
        }
    
    # Try new format with PID: fp64_monitor_<hostname>_gpu<gpu_id>_pid<pid>
    match = re.match(r'fp64_monitor_(.+)_gpu(\d+)_pid(\d+)$', basename)
    if match:
        return {
            'hostname': match.group(1),
            'gpu_id': int(match.group(2)),
            'rank': None,  # Non-MPI, use PID for uniqueness
            'pid': int(match.group(3)),
            'filename': filename
        }
    
    # Try legacy MPI format: fp64_monitor_<rank>
    match = re.match(r'fp64_monitor_(\d+)$', basename)
    if match:
        return {
            'hostname': None,  # Unknown
            'gpu_id': None,    # Unknown - will need to read from CSV
            'rank': int(match.group(1)),
            'filename': filename
        }
    
    # Try legacy non-MPI format: fp64_monitor_<hostname>_<pid>
    match = re.match(r'fp64_monitor_(.+)_(\d+)$', basename)
    if match:
        return {
            'hostname': match.group(1),
            'gpu_id': None,  # Unknown - will need to read from CSV
            'rank': None,
            'pid': int(match.group(2)),
            'filename': filename
        }
    
    # Unknown format - return minimal info
    return {
        'hostname': None,
        'gpu_id': None,
        'rank': None,
        'filename': filename
    }


_RE_STEM_GPU_PID = re.compile(r'fp64_monitor_.+_gpu(\d+)_pid(\d+)$')
_RE_STEM_GPU_RANK = re.compile(r'fp64_monitor_.+_gpu(\d+)_rank(\d+)$')
_RE_STEM_LEGACY_RANK = re.compile(r'fp64_monitor_(\d+)$')


def infer_source_layout(filenames):
    """
    Classify inputs for report wording.

    Returns one of:
        'single'     — one CSV
        'multi_gpu'  — same PID, multiple *_gpuN_pid* files (one job, many GPUs)
        'mpi'        — *_gpuN_rankR* or legacy fp64_monitor_<rank>
        'neutral'    — multiple files, layout not inferred from names
    """
    if len(filenames) <= 1:
        return 'single'
    stems = [Path(f).stem for f in filenames]
    m_pid = [_RE_STEM_GPU_PID.match(s) for s in stems]
    if all(m for m in m_pid):
        pids = {int(m.group(2)) for m in m_pid}
        gpus = {int(m.group(1)) for m in m_pid}
        if len(pids) == 1 and len(gpus) > 1:
            return 'multi_gpu'
    m_rank = [_RE_STEM_GPU_RANK.match(s) for s in stems]
    if all(m for m in m_rank):
        return 'mpi'
    if all(_RE_STEM_LEGACY_RANK.match(s) for s in stems):
        return 'mpi'
    return 'neutral'


def build_report_labels(layout):
    """Labels for console summary and fp64_report.txt (see infer_source_layout)."""
    if layout == 'multi_gpu':
        return {
            'layout': layout,
            'summary_count': 'Monitored GPUs',
            'aggregated': 'aggregated across all monitored GPUs',
            'per_source_section': 'Per-GPU summary',
            'stats_across': 'Statistical analysis across monitored GPUs',
            'no_activity': 'No active samples found in any input file.',
            'memory_scope': 'per GPU',
            'table_col': 'GPU',
            'detail_subtitle': 'Detailed per-GPU statistics',
        }
    if layout == 'mpi':
        return {
            'layout': layout,
            'summary_count': 'MPI ranks x monitored GPUs',
            'aggregated': 'aggregated across all MPI ranks x GPUs',
            'per_source_section': 'Per-MPI-GPU summary',
            'stats_across': 'Statistical analysis across MPI ranks',
            'no_activity': 'No active samples found in any MPI rank file.',
            'memory_scope': 'per MPI rank',
            'table_col': 'Rank',
            'detail_subtitle': 'Detailed per-MPI-rank statistics',
        }
    if layout == 'single':
        return {
            'layout': layout,
            'summary_count': 'Input files',
            'aggregated': '',
            'per_source_section': 'Per-file summary',
            'stats_across': 'Statistical analysis',
            'no_activity': 'No active samples found in this file.',
            'memory_scope': 'per file',
            'table_col': '#',
            'detail_subtitle': 'Detailed statistics',
        }
    return {
        'layout': layout,
        'summary_count': 'Input files',
        'aggregated': 'aggregated across all input files',
        'per_source_section': 'Per-file summary',
        'stats_across': 'Statistical analysis across input files',
        'no_activity': 'No active samples found in any input file.',
        'memory_scope': 'per input file',
        'table_col': 'Src',
        'detail_subtitle': 'Detailed per-file statistics',
    }


def section_title(base, labels, n_sources):
    """Append aggregation phrase only when multiple sources."""
    agg = labels.get('aggregated') or ''
    if n_sources > 1 and agg:
        return f"{base} ({agg})"
    return base


def source_table_label(filename, layout, fallback_index):
    """First-column label for per-source summary table."""
    info = parse_filename_info(filename)
    if layout == 'multi_gpu' and info.get('gpu_id') is not None:
        return str(info['gpu_id'])
    if layout == 'mpi' and info.get('rank') is not None:
        return str(info['rank'])
    return str(fallback_index)


def format_source_totals_note(valid, total, labels):
    """Footnote for memory HWM totals, e.g. '4 GPUs' or '2 of 4 MPI rank files'."""
    layout = labels['layout']
    if layout == 'multi_gpu':
        if valid == total:
            return f"{valid} GPUs"
        return f"{valid} of {total} GPU files"
    if layout == 'mpi':
        if valid == total:
            return f"{valid} MPI ranks"
        return f"{valid} of {total} MPI rank x GPU files"
    if valid == total:
        return f"{valid} files"
    return f"{valid} of {total} files"


def get_gpu_id_from_csv(filename):
    """Read GPU ID from the first data row of a CSV file."""
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                # First non-comment line is header
                reader = csv.DictReader([line] + f.readlines())
                for row in reader:
                    gpu_id = row.get('gpu_id')
                    if gpu_id is not None:
                        return int(gpu_id)
                    break
    except Exception:
        pass
    return 0  # Default to GPU 0


def deduplicate_files(filenames):
    """
    Deduplicate files to keep only one per (hostname, gpu_id) combination.
    
    When multiple processes on the same host monitor the same GPU, the hardware
    counters are shared (not per-process). We should only include one file per
    (hostname, gpu_id) to avoid double-counting.
    
    Returns: (deduplicated_files, excluded_files, dedup_info)
    """
    # Parse all filenames
    file_info = []
    for f in filenames:
        info = parse_filename_info(f)
        # If GPU ID not in filename, try to read from CSV
        if info['gpu_id'] is None:
            info['gpu_id'] = get_gpu_id_from_csv(f)
        # If hostname not in filename, use 'unknown'
        if info['hostname'] is None:
            info['hostname'] = 'unknown'
        file_info.append(info)
    
    # Group by (hostname, gpu_id)
    groups = defaultdict(list)
    for info in file_info:
        key = (info['hostname'], info['gpu_id'])
        groups[key].append(info)
    
    # For each group, keep the file with the lowest rank (or first if no rank)
    deduplicated = []
    excluded = []
    dedup_info = []
    
    for (hostname, gpu_id), infos in sorted(groups.items()):
        if len(infos) == 1:
            # Only one file for this (hostname, gpu_id) - keep it
            deduplicated.append(infos[0]['filename'])
        else:
            # Multiple files - keep the one with lowest rank
            # Sort by rank (None ranks go to end)
            sorted_infos = sorted(infos, key=lambda x: (x.get('rank') is None, x.get('rank', 0)))
            kept = sorted_infos[0]
            deduplicated.append(kept['filename'])
            
            # Track excluded files
            for info in sorted_infos[1:]:
                excluded.append(info['filename'])
            
            # Record deduplication info
            dedup_info.append({
                'hostname': hostname,
                'gpu_id': gpu_id,
                'kept': kept['filename'],
                'excluded': [i['filename'] for i in sorted_infos[1:]]
            })
    
    return deduplicated, excluded, dedup_info


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
    """Write detailed report with per-source statistics to file."""

    layout = infer_source_layout(filenames)
    labels = build_report_labels(layout)

    # Load data per source file
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
            f.write(labels['no_activity'] + "\n")
            return
        
        # Per-source summary table
        f.write("-" * 78 + "\n")
        f.write(labels['per_source_section'] + "\n")
        f.write("-" * 78 + "\n\n")

        col = labels['table_col']
        f.write(f"{col:<8} {'Samples':>8} {'FP64 VALU':>15} {'Total VALU':>15} {'FP64%':>8} {'GPU HWM':>10}\n")
        f.write("-" * 78 + "\n")

        for i, filename in enumerate(sorted(rank_totals.keys())):
            row_label = source_table_label(filename, layout, i)
            data = rank_totals[filename]
            f.write(f"{row_label:<8} {data['active_samples']:>8} {data['fp64_valu']:>15,.0f} "
                   f"{data['total_valu']:>15,.0f} {data['fp64_valu_pct']:>7.2f}% "
                   f"{data['gpu_hwm']:>9.1f}M\n")
        f.write("\n")

        f.write("-" * 78 + "\n")
        f.write(labels['stats_across'] + "\n")
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
    """Write detailed per-source statistics to an already open file handle."""

    layout = infer_source_layout(filenames)
    labels = build_report_labels(layout)

    # Load data per source file
    rank_data = {}
    for filename in filenames:
        rows, _ = load_fp64_monitor_csv(filename)
        if rows:
            rank_data[filename] = rows
    
    if not rank_data:
        return
    
    num_ranks = len(rank_data)
    
    f.write("=" * 78 + "\n")
    f.write(labels['detail_subtitle'] + "\n")
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
        f.write(labels['no_activity'] + "\n")
        return

    f.write("-" * 78 + "\n")
    f.write(labels['per_source_section'] + "\n")
    f.write("-" * 78 + "\n\n")

    col = labels['table_col']
    f.write(f"{col:<8} {'Samples':>8} {'FP64 VALU':>15} {'Total VALU':>15} {'FP64%':>8} {'GPU HWM':>10}\n")
    f.write("-" * 78 + "\n")

    for i, filename in enumerate(sorted(rank_totals.keys())):
        row_label = source_table_label(filename, layout, i)
        data = rank_totals[filename]
        f.write(f"{row_label:<8} {data['active_samples']:>8} {data['fp64_valu']:>15,.0f} "
               f"{data['total_valu']:>15,.0f} {data['fp64_valu_pct']:>7.2f}% "
               f"{data['gpu_hwm']:>9.1f}M\n")
    f.write("\n")

    f.write("-" * 78 + "\n")
    f.write(labels['stats_across'] + "\n")
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


def detect_counter_set(rows):
    """Detect which counter set was used from CSV data."""
    if not rows:
        return "unknown"
    
    # Check if counter_set column exists and use it
    if 'counter_set' in rows[0]:
        counter_sets = set(r.get('counter_set', '') for r in rows if r.get('counter_set'))
        if counter_sets:
            # Return the most common one (or first if all same)
            return list(counter_sets)[0]
    
    # Fallback: infer from which counters have non-zero values
    sample = rows[0]
    
    # Check for highlevel-specific counters
    has_highlevel = any(float(sample.get(c, 0)) > 0 for c in 
                        ['valu_instructions', 'salu_total', 'vmem_total', 'smem_total', 
                         'lds_total', 'branch_total', 'mfma_total'])
    
    # Check for FP64-specific counters
    has_fp64 = any(float(sample.get(c, 0)) > 0 for c in 
                   ['fp64_add', 'fp64_mul', 'fp64_fma', 'fp64_trans', 'fp64_mfma'])
    
    # Check for FP32-specific counters
    has_fp32 = any(float(sample.get(c, 0)) > 0 for c in 
                   ['fp32_add', 'fp32_mul', 'fp32_fma', 'fp32_trans', 'fp32_mfma'])
    
    # Check for highlevel2-specific counters
    has_highlevel2 = any(float(sample.get(c, 0)) > 0 for c in 
                         ['flat_total', 'gds_total', 'exp_gds_total', 
                          'sendmsg_total', 'vskipped_total'])
    
    if has_fp64 and not has_fp32:
        return "fp64"
    elif has_fp32:
        return "fp32"
    elif has_highlevel2 and not has_highlevel:
        return "highlevel2"
    elif has_highlevel:
        return "highlevel"
    else:
        return "unknown"


def analyze_fp64_monitor_files(filenames):
    """Analyze and aggregate CSV output from one or more fp64_monitor files."""

    all_rows = []

    for filename in filenames:
        rows, _ = load_fp64_monitor_csv(filename)
        for row in rows:
            row['_source_file'] = filename
        all_rows.extend(rows)

    if not all_rows:
        print("Error: No data found in CSV files")
        return

    layout = infer_source_layout(filenames)
    labels = build_report_labels(layout)
    num_sources = len(filenames)

    # Detect counter set
    counter_set = detect_counter_set(all_rows)
    
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
        p(f"Counter Set:   {counter_set}")
        p(f"{(labels['summary_count'] + ':'):<17} {num_sources}")
        p(f"Total samples: {len(all_rows)}")

        if num_sources > 1:
            if layout == 'multi_gpu':
                p("(Each file is one GPU from the same process; totals sum counters across GPUs.)")
            elif layout == 'mpi':
                p("(Each file is one MPI rank x one GPU; totals sum counters across GPUs.)")
            else:
                p("(Totals sum each file's final counter maxima; MPI vs multi-GPU not inferred from names.)")
            p()

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
                """Sum the final (max) value of a counter per input file, across files."""
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
                if total_valu > 0:
                    p(f"  Total VALU:      {total_valu:>18,.0f}")
                p(f"  Total Instr:     {total_insts:>18,.0f}")
                p()
            
            # High-Level Instruction Mix - ONLY show if highlevel counter set was used
            if counter_set == 'highlevel':
                total_salu = sum_across_ranks('salu_total')
                total_vmem = sum_across_ranks('vmem_total')
                total_smem = sum_across_ranks('smem_total')
                total_lds = sum_across_ranks('lds_total')
                total_branch = sum_across_ranks('branch_total')
                total_mfma = sum_across_ranks('mfma_total')
                
                highlevel_sum = total_valu + total_salu + total_vmem + total_smem + total_lds + total_branch + total_mfma
                if highlevel_sum > 0 and total_insts > 0:
                    p("-" * 70)
                    p(section_title("High-Level Instruction Mix", labels, num_sources))
                    p("-" * 70)
                    p()
                    p(f"  {'Category':<12} {'Count':>18} {'% of Total':>12}")
                    p(f"  {'-'*12} {'-'*18} {'-'*12}")
                    p(f"  {'VALU':<12} {total_valu:>18,.0f} {100*total_valu/total_insts:>11.2f}%")
                    p(f"  {'SALU':<12} {total_salu:>18,.0f} {100*total_salu/total_insts:>11.2f}%")
                    p(f"  {'VMEM':<12} {total_vmem:>18,.0f} {100*total_vmem/total_insts:>11.2f}%")
                    p(f"  {'SMEM':<12} {total_smem:>18,.0f} {100*total_smem/total_insts:>11.2f}%")
                    p(f"  {'LDS':<12} {total_lds:>18,.0f} {100*total_lds/total_insts:>11.2f}%")
                    p(f"  {'Branch':<12} {total_branch:>18,.0f} {100*total_branch/total_insts:>11.2f}%")
                    p(f"  {'MFMA':<12} {total_mfma:>18,.0f} {100*total_mfma/total_insts:>11.2f}%")
                    p(f"  {'-'*12} {'-'*18} {'-'*12}")
                    p(f"  {'Subtotal':<12} {highlevel_sum:>18,.0f} {100*highlevel_sum/total_insts:>11.2f}%")
                    p(f"  {'Total':<12} {total_insts:>18,.0f} {'100.00%':>12}")
                    p()
            
            # High-Level 2: Other instruction types - ONLY show if highlevel2 counter set was used
            if counter_set == 'highlevel2':
                total_flat = sum_across_ranks('flat_total')
                total_gds = sum_across_ranks('gds_total')
                total_exp_gds = sum_across_ranks('exp_gds_total')
                total_sendmsg = sum_across_ranks('sendmsg_total')
                total_vskipped = sum_across_ranks('vskipped_total')
                
                highlevel2_sum = total_flat + total_gds + total_exp_gds + total_sendmsg + total_vskipped
                if highlevel2_sum > 0 and total_insts > 0:
                    p("-" * 70)
                    p(section_title("Other Instruction Types", labels, num_sources))
                    p("-" * 70)
                    p()
                    p(f"  {'Category':<12} {'Count':>18} {'% of Total':>12}")
                    p(f"  {'-'*12} {'-'*18} {'-'*12}")
                    p(f"  {'FLAT':<12} {total_flat:>18,.0f} {100*total_flat/total_insts:>11.2f}%")
                    p(f"  {'GDS':<12} {total_gds:>18,.0f} {100*total_gds/total_insts:>11.2f}%")
                    p(f"  {'EXP_GDS':<12} {total_exp_gds:>18,.0f} {100*total_exp_gds/total_insts:>11.2f}%")
                    p(f"  {'SENDMSG':<12} {total_sendmsg:>18,.0f} {100*total_sendmsg/total_insts:>11.2f}%")
                    p(f"  {'VSKIPPED':<12} {total_vskipped:>18,.0f} {100*total_vskipped/total_insts:>11.2f}%")
                    p(f"  {'-'*12} {'-'*18} {'-'*12}")
                    p(f"  {'Subtotal':<12} {highlevel2_sum:>18,.0f} {100*highlevel2_sum/total_insts:>11.2f}%")
                    p()
            
            # FP64 Instruction Counts - show if fp64 counter set was used or data present
            total_fp64_add = sum_across_ranks('fp64_add')
            total_fp64_mul = sum_across_ranks('fp64_mul')
            total_fp64_fma = sum_across_ranks('fp64_fma')
            total_fp64_trans = sum_across_ranks('fp64_trans')
            total_fp64_mfma = sum_across_ranks('fp64_mfma')
            total_fp64_valu = total_fp64_add + total_fp64_mul + total_fp64_fma + total_fp64_trans + total_fp64_mfma
            
            # Only show FP64 section if fp64 counter set was used
            if counter_set == 'fp64' and total_fp64_valu > 0:
                p("-" * 70)
                p(section_title("FP64 Instruction Counts", labels, num_sources))
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
            
            # FP32 Instruction Counts - show only if fp32 counter set was used
            if counter_set == 'fp32':
                total_fp32_add = sum_across_ranks('fp32_add')
                total_fp32_mul = sum_across_ranks('fp32_mul')
                total_fp32_fma = sum_across_ranks('fp32_fma')
                total_fp32_trans = sum_across_ranks('fp32_trans')
                total_fp32_mfma = sum_across_ranks('fp32_mfma')
                total_fp32_valu = total_fp32_add + total_fp32_mul + total_fp32_fma + total_fp32_trans + total_fp32_mfma
                
                if total_fp32_valu > 0:
                    p("-" * 70)
                    p(section_title("FP32 Instruction Counts", labels, num_sources))
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
            
            # FP16/BF16 Instruction Counts - show only if fp16 counter set was used
            if counter_set == 'fp16':
                total_fp16_add = sum_across_ranks('fp16_add')
                total_fp16_mul = sum_across_ranks('fp16_mul')
                total_fp16_fma = sum_across_ranks('fp16_fma')
                total_fp16_mfma = sum_across_ranks('fp16_mfma')
                total_bf16_mfma = sum_across_ranks('bf16_mfma')
                total_fp16_valu = total_fp16_add + total_fp16_mul + total_fp16_fma + total_fp16_mfma + total_bf16_mfma
                
                if total_fp16_valu > 0:
                    p("-" * 70)
                    p(section_title("FP16/BF16 Instruction Counts", labels, num_sources))
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
            
            # Integer Instruction Counts - show only if int counter set was used
            if counter_set == 'int':
                total_int32 = sum_across_ranks('int32_total')
                total_int64 = sum_across_ranks('int64_total')
                total_int_valu = total_int32 + total_int64
                
                if total_int_valu > 0:
                    p("-" * 70)
                    p(section_title("Integer Instruction Counts", labels, num_sources))
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
            
            # SALU / Scalar Instructions - show only if salu counter set was used
            if counter_set == 'salu':
                total_salu_c = sum_across_ranks('salu_total')
                total_smem_c = sum_across_ranks('smem_total')
                total_flat_c = sum_across_ranks('flat_total')
                total_lds_c = sum_across_ranks('lds_total')
                total_branch_c = sum_across_ranks('branch_total')
                total_scalar = total_salu_c + total_smem_c
                
                if total_salu_c > 0 or total_smem_c > 0:
                    p("-" * 70)
                    p(section_title("Scalar & Memory Instructions", labels, num_sources))
                    p("-" * 70)
                    p()
                    p(f"  SALU (Scalar ALU): {total_salu_c:>18,.0f}")
                    p(f"  SMEM (Scalar Mem): {total_smem_c:>18,.0f}")
                    p(f"  FLAT:              {total_flat_c:>18,.0f}")
                    p(f"  LDS:               {total_lds_c:>18,.0f}")
                    p(f"  Branch:            {total_branch_c:>18,.0f}")
                    p(f"  ----------------------------------------")
                    p(f"  Total SALU+SMEM:   {total_scalar:>18,.0f}")
                    p()
                    if total_insts > 0:
                        p(f"SALU %% of Total:    {(total_salu_c / total_insts) * 100:>10.2f}%")
                        p(f"SMEM %% of Total:    {(total_smem_c / total_insts) * 100:>10.2f}%")
                        p(f"FLAT %% of Total:    {(total_flat_c / total_insts) * 100:>10.2f}%")
                        p(f"LDS %% of Total:     {(total_lds_c / total_insts) * 100:>10.2f}%")
                        p(f"Branch %% of Total:  {(total_branch_c / total_insts) * 100:>10.2f}%")
                    p()
        
        # Memory Analysis - compute per-file HWMs then aggregate
        # Check for memory data - look for vram_hwm_mb (amd-smi) or cpu_resident_hwm_mb
        has_memory_data = (
            'vram_hwm_mb' in all_rows[0] or 
            'gtt_hwm_mb' in all_rows[0] or
            'cpu_resident_hwm_mb' in all_rows[0]
        ) if all_rows else False
        
        if has_memory_data:
            rank_rows = defaultdict(list)
            for row in all_rows:
                rank_rows[row.get('_source_file', 'unknown')].append(row)
            
            total_sources = len(rank_rows)

            per_rank_vram_hwm = []
            per_rank_gtt_hwm = []
            per_rank_cpu_hwm = []
            vram_total_per_rank = 0
            
            for filename, rows in rank_rows.items():
                vram_hwm = max((float(r.get('vram_hwm_mb', 0)) for r in rows), default=0)
                gtt_hwm = max((float(r.get('gtt_hwm_mb', 0)) for r in rows), default=0)
                cpu_hwm = max((float(r.get('cpu_resident_hwm_mb', 0)) for r in rows), default=0)
                
                if vram_hwm > 0:
                    per_rank_vram_hwm.append(vram_hwm)
                if gtt_hwm > 0:
                    per_rank_gtt_hwm.append(gtt_hwm)
                if cpu_hwm > 0:
                    per_rank_cpu_hwm.append(cpu_hwm)
                
                vram_totals = [float(r.get('vram_total_mb', 0)) for r in rows if float(r.get('vram_total_mb', 0)) > 0]
                if vram_totals:
                    vram_total_per_rank = vram_totals[0]
            
            has_any_memory = per_rank_vram_hwm or per_rank_gtt_hwm or per_rank_cpu_hwm
            if has_any_memory:
                p("-" * 70)
                p(section_title("Memory Usage", labels, num_sources))
                p("-" * 70)
                p()
                
                if per_rank_vram_hwm:
                    stats = compute_stats(per_rank_vram_hwm)
                    total_hwm = sum(per_rank_vram_hwm)
                    scope = labels['memory_scope']
                    p(f"VRAM (GPU Memory): {vram_total_per_rank/1024:.1f} GB {scope}")
                    p(f"  HWM Min:    {stats['min']:>10.1f} MB")
                    p(f"  HWM Max:    {stats['max']:>10.1f} MB")
                    p(f"  HWM Mean:   {stats['mean']:>10.1f} MB")
                    p(f"  HWM StdDev: {stats['stdev']:>10.2f} MB")
                    note = format_source_totals_note(len(per_rank_vram_hwm), total_sources, labels)
                    p(f"  HWM Total:  {total_hwm/1024:>10.2f} GB ({note})")
                    p()
                
                if per_rank_gtt_hwm:
                    stats = compute_stats(per_rank_gtt_hwm)
                    total_hwm = sum(per_rank_gtt_hwm)
                    p(f"GTT (system memory for GPU):")
                    p(f"  HWM Min:    {stats['min']:>10.1f} MB")
                    p(f"  HWM Max:    {stats['max']:>10.1f} MB")
                    p(f"  HWM Mean:   {stats['mean']:>10.1f} MB")
                    p(f"  HWM StdDev: {stats['stdev']:>10.2f} MB")
                    note = format_source_totals_note(len(per_rank_gtt_hwm), total_sources, labels)
                    p(f"  HWM Total:  {total_hwm/1024:>10.2f} GB ({note})")
                    p()
                
                if per_rank_cpu_hwm:
                    stats = compute_stats(per_rank_cpu_hwm)
                    total_hwm = sum(per_rank_cpu_hwm)
                    p(f"CPU Resident Memory:")
                    p(f"  HWM Min:    {stats['min']:>10.1f} MB")
                    p(f"  HWM Max:    {stats['max']:>10.1f} MB")
                    p(f"  HWM Mean:   {stats['mean']:>10.1f} MB")
                    p(f"  HWM StdDev: {stats['stdev']:>10.2f} MB")
                    note = format_source_totals_note(len(per_rank_cpu_hwm), total_sources, labels)
                    p(f"  HWM Total:  {total_hwm/1024:>10.2f} GB ({note})")
                    p()
        
        p("=" * 70)
        
        # Detailed per-source report (file only, not stdout)
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
        print("Usage: fp64_monitor_analyze_results.py <csv_file> [csv_file2 ...]")
        print("       fp64_monitor_analyze_results.py fp64_monitor_*.csv")
        print()
        print("When multiple files are provided, data is aggregated (see script docstring).")
        print("Files from the same (hostname, GPU) are deduplicated to avoid double-counting.")
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
        # Deduplicate files by (hostname, gpu_id) to avoid double-counting
        # when multiple processes on the same host monitor the same GPU
        deduplicated, excluded, dedup_info = deduplicate_files(filenames)
        
        if excluded:
            print("=" * 70)
            print("File Deduplication")
            print("=" * 70)
            print()
            print(f"Input files:       {len(filenames)}")
            print(f"After dedup:       {len(deduplicated)}")
            print(f"Excluded:          {len(excluded)}")
            print()
            print("Hardware counters are per-GPU, not per-process. When multiple")
            print("processes monitor the same GPU, only one file is used per")
            print("(hostname, GPU) to avoid double-counting.")
            print()
            for info in dedup_info:
                print(f"  {info['hostname']} GPU {info['gpu_id']}:")
                print(f"    Kept:     {Path(info['kept']).name}")
                for ex in info['excluded']:
                    print(f"    Excluded: {Path(ex).name}")
            print()
        
        analyze_fp64_monitor_files(deduplicated)
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
