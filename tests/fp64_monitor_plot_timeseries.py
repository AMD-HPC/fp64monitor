#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE in the project root for license information.

"""
Time-series analysis of FP64 Monitor CSV output.

Generates charts showing instruction mix over time:
  1. Raw instruction counts per sample period
  2. Percentage of VALU per sample period

Supports multiple counter sets (default: fp64):
  - FP64: fp64_add, fp64_mul, fp64_fma, fp64_trans, fp64_mfma
  - FP32: fp32_add, fp32_mul, fp32_fma, fp32_trans, fp32_mfma
  - INT:  int32_total, int64_total, vmem_rd, vmem_wr

Use --counters auto to auto-detect based on input data.

When multiple CSV files are provided (e.g., from MPI ranks), the data is
aggregated (summed) across all files for each sample period.

Output formats:
  - PDF (default): Static charts using matplotlib
  - HTML (--html): Interactive charts using plotly (zoom, hover, inspect)

Requirements:
    - Python 3.8+
    - For PDF: pip install matplotlib
    - For HTML: pip install plotly

Usage:
    python3 fp64_monitor_plot_timeseries.py <csv_files...> [--output prefix] [--html]
    python3 fp64_monitor_plot_timeseries.py fp64_monitor_*.csv --output my_analysis
    python3 fp64_monitor_plot_timeseries.py fp64_monitor_*.csv --html
    python3 fp64_monitor_plot_timeseries.py fp64_monitor_*.csv --counters int  # Use INT counters
    python3 fp64_monitor_plot_timeseries.py fp64_monitor_*.csv -c auto         # Auto-detect
"""

import sys
import csv
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for PDF generation
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Counter set definitions
COUNTER_SETS = {
    'fp64': {
        'name': 'FP64',
        'columns': ['fp64_add', 'fp64_mul', 'fp64_fma', 'fp64_trans', 'fp64_mfma'],
        'labels': ['FP64 ADD', 'FP64 MUL', 'FP64 FMA', 'FP64 Trans', 'FP64 MFMA'],
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    },
    'fp32': {
        'name': 'FP32',
        'columns': ['fp32_add', 'fp32_mul', 'fp32_fma', 'fp32_trans', 'fp32_mfma'],
        'labels': ['FP32 ADD', 'FP32 MUL', 'FP32 FMA', 'FP32 Trans', 'FP32 MFMA'],
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    },
    'int': {
        'name': 'Integer + VMEM',
        'columns': ['int32_total', 'int64_total', 'vmem_rd', 'vmem_wr'],
        'labels': ['INT32', 'INT64', 'VMEM Read', 'VMEM Write'],
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    },
    'salu': {
        'name': 'SALU + Memory',
        'columns': ['salu_total', 'smem_total', 'flat_total', 'lds_total', 'branch_total'],
        'labels': ['SALU', 'SMEM', 'FLAT', 'LDS', 'Branch'],
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    },
    'highlevel': {
        'name': 'High-Level Instruction Mix',
        'columns': ['valu_instructions', 'salu_total', 'vmem_total', 'smem_total', 'lds_total', 'branch_total', 'mfma_total'],
        'labels': ['VALU', 'SALU', 'VMEM', 'SMEM', 'LDS', 'Branch', 'MFMA'],
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'],
    },
    'highlevel2': {
        'name': 'Other Instruction Types',
        'columns': ['flat_total', 'gds_total', 'exp_gds_total', 'sendmsg_total', 'vskipped_total'],
        'labels': ['FLAT', 'GDS', 'EXP_GDS', 'SENDMSG', 'VSKIPPED'],
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    },
}


def load_csv(filename):
    """Load CSV data from fp64_monitor output file."""
    samples = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if not line.strip():
                continue
            samples.append(line)
    
    if not samples:
        return []
    
    reader = csv.DictReader(samples)
    return list(reader)


def detect_counter_set(rows):
    """Detect which counter set has non-zero data."""
    if not rows:
        return 'fp64'  # Default
    
    # Sum values across all rows to detect which set has data
    totals = {}
    for set_name, set_info in COUNTER_SETS.items():
        total = 0
        for row in rows:
            for col in set_info['columns']:
                total += float(row.get(col, 0))
        totals[set_name] = total
    
    # Return the set with the most data
    best_set = max(totals, key=totals.get)
    if totals[best_set] > 0:
        return best_set
    return 'fp64'  # Default if all zeros


def get_counter_set(csv_files, specified_set):
    """Get the counter set to use, with fp64 as default."""
    if specified_set == 'auto':
        rows = load_csv(csv_files[0])
        detected = detect_counter_set(rows)
        print(f"Auto-detected counter set: {detected}")
        return detected
    elif specified_set is not None:
        return specified_set
    else:
        return 'fp64'  # Default


def compute_deltas(rows, counter_set='fp64'):
    """Compute per-period deltas from cumulative counter values."""
    if not rows:
        return []
    
    set_info = COUNTER_SETS[counter_set]
    columns = set_info['columns']
    
    deltas = []
    prev = None
    
    for row in rows:
        current = {
            'sample_id': int(row.get('sample_id', 0)),
            'elapsed_seconds': float(row.get('elapsed_seconds', 0)),
            'total_instructions': float(row.get('total_instructions', 0)),
            'valu_instructions': float(row.get('valu_instructions', 0)),
        }
        # Add counter-set specific columns
        for col in columns:
            current[col] = float(row.get(col, 0))
        
        if prev is None:
            # First sample - use raw values
            delta = current.copy()
        else:
            # Subsequent samples - compute delta
            delta = {
                'sample_id': current['sample_id'],
                'elapsed_seconds': current['elapsed_seconds'],
                'total_instructions': current['total_instructions'] - prev['total_instructions'],
                'valu_instructions': current['valu_instructions'] - prev['valu_instructions'],
            }
            for col in columns:
                delta[col] = current[col] - prev[col]
        
        deltas.append(delta)
        prev = current
    
    return deltas


def load_and_aggregate(csv_files, counter_set='fp64'):
    """Load data from multiple CSV files and aggregate by sample period.
    
    Returns aggregated data (averages across ranks) with min/max for variability bands.
    """
    set_info = COUNTER_SETS[counter_set]
    columns = set_info['columns']
    
    # Load deltas from each file
    all_deltas = []
    for csv_file in csv_files:
        rows = load_csv(csv_file)
        if rows:
            deltas = compute_deltas(rows, counter_set)
            all_deltas.append(deltas)
    
    if not all_deltas:
        return []
    
    # Find the maximum number of samples across all files
    max_samples = max(len(d) for d in all_deltas)
    
    # Aggregate by sample_id (average across all ranks, track min/max per rank)
    aggregated = []
    for i in range(max_samples):
        # Collect per-rank values for this sample period
        rank_total_insts = []
        rank_valu_insts = []
        
        agg = {
            'sample_id': i + 1,
            'elapsed_seconds': 0,
            'total_instructions': 0,
            'valu_instructions': 0,
        }
        for col in columns:
            agg[col] = 0
        
        count = 0
        for deltas in all_deltas:
            if i < len(deltas):
                d = deltas[i]
                agg['total_instructions'] += d['total_instructions']
                agg['valu_instructions'] += d['valu_instructions']
                for col in columns:
                    agg[col] += d[col]
                agg['elapsed_seconds'] = max(agg['elapsed_seconds'], d['elapsed_seconds'])
                rank_total_insts.append(d['total_instructions'])
                rank_valu_insts.append(d['valu_instructions'])
                count += 1
        
        if count > 0:
            # Convert sums to averages
            if count > 1:
                agg['total_instructions'] /= count
                agg['valu_instructions'] /= count
                for col in columns:
                    agg[col] /= count
            
            # Store min/max across ranks (actual per-rank values, not scaled)
            agg['total_insts_min'] = min(rank_total_insts) if rank_total_insts else 0
            agg['total_insts_max'] = max(rank_total_insts) if rank_total_insts else 0
            agg['valu_insts_min'] = min(rank_valu_insts) if rank_valu_insts else 0
            agg['valu_insts_max'] = max(rank_valu_insts) if rank_valu_insts else 0
            aggregated.append(agg)
    
    return aggregated


def plot_raw_counts(ax, deltas, counter_set='fp64'):
    """Plot raw instruction counts as lines + stacked area."""
    if not deltas:
        return
    
    set_info = COUNTER_SETS[counter_set]
    columns = set_info['columns']
    labels = set_info['labels']
    colors = set_info['colors']
    set_name = set_info['name']
    
    x = [d['sample_id'] for d in deltas]
    total_insts = [d['total_instructions'] for d in deltas]
    valu_insts = [d['valu_instructions'] for d in deltas]
    
    # Extract counter-specific data
    counter_data = []
    for col in columns:
        counter_data.append([d[col] for d in deltas])
    
    # Check if we have variability data (from multi-rank aggregation)
    has_variability = 'total_insts_min' in deltas[0]
    
    # Stacked area for counter components
    ax.stackplot(x, *counter_data, labels=labels, colors=colors, alpha=0.7)
    
    # Variability bands (if multi-rank data)
    if has_variability:
        total_min = [d['total_insts_min'] for d in deltas]
        total_max = [d['total_insts_max'] for d in deltas]
        valu_min = [d['valu_insts_min'] for d in deltas]
        valu_max = [d['valu_insts_max'] for d in deltas]
        
        ax.fill_between(x, total_min, total_max, alpha=0.2, color='black', 
                        label='Total Instr (rank variability)')
        ax.fill_between(x, valu_min, valu_max, alpha=0.2, color='gray',
                        label='Total VALU (rank variability)')
    
    # Lines for totals
    ax.plot(x, total_insts, 'k-', linewidth=2, label='Total Instructions')
    ax.plot(x, valu_insts, 'k--', linewidth=2, label='Total VALU')
    
    ax.set_xlabel('Sample Period')
    ax.set_ylabel('Instruction Count')
    ax.set_title(f'Instruction Mix Over Time (Raw Counts) - {set_name}')
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis with commas for thousands
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )


def plot_percentage(ax, deltas, counter_set='fp64'):
    """Plot counter components as percentage of VALU (or Total for highlevel sets)."""
    if not deltas:
        return
    
    set_info = COUNTER_SETS[counter_set]
    columns = set_info['columns']
    labels = set_info['labels']
    colors = set_info['colors']
    set_name = set_info['name']
    
    # For highlevel sets, use total_instructions as denominator
    use_total = counter_set in ('highlevel', 'highlevel2')
    denom_label = 'Total' if use_total else 'VALU'
    
    x = [d['sample_id'] for d in deltas]
    
    # Compute percentages (handle division by zero)
    pct_data = [[] for _ in columns]
    
    for d in deltas:
        denom = d['total_instructions'] if use_total else d['valu_instructions']
        for i, col in enumerate(columns):
            if denom > 0:
                pct_data[i].append(100.0 * d[col] / denom)
            else:
                pct_data[i].append(0)
    
    # Stacked area for percentage
    ax.stackplot(x, *pct_data, labels=labels, colors=colors, alpha=0.7)
    
    ax.set_xlabel('Sample Period')
    ax.set_ylabel(f'Percentage of {denom_label} (%)')
    ax.set_title(f'{set_name} Instruction Mix Over Time (% of {denom_label})')
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)


def generate_plots(csv_files, output_prefix=None, counter_set=None):
    """Generate PDF with time-series plots."""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for plotting.")
        print("Install with: pip install matplotlib")
        return None
    
    # Handle single file or list
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    
    # Get counter set (fp64 is default)
    counter_set = get_counter_set(csv_files, counter_set)
    
    # Load and aggregate data from all files
    if len(csv_files) == 1:
        rows = load_csv(csv_files[0])
        if not rows:
            print(f"Error: No data found in {csv_files[0]}")
            return None
        deltas = compute_deltas(rows, counter_set)
    else:
        deltas = load_and_aggregate(csv_files, counter_set)
    
    if not deltas:
        print("Error: Could not compute deltas from input files")
        return None
    
    # Determine output filename
    if output_prefix is None:
        output_prefix = "fp64_monitor"
    output_file = f"{output_prefix}_timeseries.pdf"
    
    set_name = COUNTER_SETS[counter_set]['name']
    
    # Create title
    if len(csv_files) == 1:
        title = f'{set_name} Monitor Time-Series Analysis\n{Path(csv_files[0]).name}'
    else:
        title = f'{set_name} Monitor Time-Series Analysis\n({len(csv_files)} files aggregated)'
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Plot raw counts (full timeline including idle periods)
    plot_raw_counts(ax1, deltas, counter_set)
    
    # Plot percentages
    plot_percentage(ax2, deltas, counter_set)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save to PDF
    plt.savefig(output_file, format='pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated: {output_file}")
    return output_file


def generate_html_plots(csv_files, output_prefix=None, counter_set=None):
    """Generate interactive HTML with time-series plots using plotly."""
    if not HAS_PLOTLY:
        print("Error: plotly is required for HTML output.")
        print("Install with: pip install plotly")
        return None
    
    # Handle single file or list
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    
    # Get counter set (fp64 is default)
    counter_set = get_counter_set(csv_files, counter_set)
    
    set_info = COUNTER_SETS[counter_set]
    columns = set_info['columns']
    labels = set_info['labels']
    colors_list = set_info['colors']
    set_name = set_info['name']
    
    # Load and aggregate data from all files
    if len(csv_files) == 1:
        rows = load_csv(csv_files[0])
        if not rows:
            print(f"Error: No data found in {csv_files[0]}")
            return None
        deltas = compute_deltas(rows, counter_set)
    else:
        deltas = load_and_aggregate(csv_files, counter_set)
    
    if not deltas:
        print("Error: Could not compute deltas from input files")
        return None
    
    # Determine output filename
    if output_prefix is None:
        output_prefix = "fp64_monitor"
    output_file = f"{output_prefix}_timeseries.html"
    
    num_files = len(csv_files)
    
    # Extract data series (full timeline including idle periods)
    x = [d['sample_id'] for d in deltas]
    elapsed = [d['elapsed_seconds'] for d in deltas]
    total_insts = [d['total_instructions'] for d in deltas]
    valu_insts = [d['valu_instructions'] for d in deltas]
    
    # Extract counter-specific data
    counter_data = {}
    for col in columns:
        counter_data[col] = [d[col] for d in deltas]
    
    # Check if we have variability data (from multi-rank aggregation)
    has_variability = 'total_insts_min' in deltas[0]
    if has_variability:
        total_min = [d['total_insts_min'] for d in deltas]
        total_max = [d['total_insts_max'] for d in deltas]
        valu_min = [d['valu_insts_min'] for d in deltas]
        valu_max = [d['valu_insts_max'] for d in deltas]
    
    # For highlevel sets, use total_instructions as denominator
    use_total = counter_set in ('highlevel', 'highlevel2')
    denom_label = 'Total' if use_total else 'VALU'
    
    # Compute percentages
    pct_data = {col: [] for col in columns}
    for d in deltas:
        denom = d['total_instructions'] if use_total else d['valu_instructions']
        for col in columns:
            if denom > 0:
                pct_data[col].append(100.0 * d[col] / denom)
            else:
                pct_data[col].append(0)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Instruction Mix Over Time (Raw Counts) - {set_name}',
            f'{set_name} Instruction Mix Over Time (% of {denom_label})'
        ),
        vertical_spacing=0.12
    )
    
    # Chart 1: Raw counts - stacked area for counter components
    for i, (col, label, color) in enumerate(zip(columns, labels, colors_list)):
        fill = 'tozeroy' if i == 0 else 'tonexty'
        fig.add_trace(go.Scatter(
            x=x, y=counter_data[col], name=label, fill=fill,
            mode='lines', line=dict(width=0.5, color=color),
            stackgroup='counters', groupnorm='',
            hovertemplate=f'Sample %{{x}}<br>{label}: %{{y:,.0f}}<extra></extra>'
        ), row=1, col=1)
    
    # Chart 1: Variability bands (if multi-rank data)
    if has_variability:
        # Total Instructions variability band
        fig.add_trace(go.Scatter(
            x=x, y=total_max, name='Total Instr (max if all ranks had max)',
            mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
            showlegend=False,
            hovertemplate='Sample %{x}<br>Total Instr (max): %{y:,.0f}<extra></extra>'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=total_min, name='Total Instr variability',
            mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
            fill='tonexty', fillcolor='rgba(0,0,0,0.15)',
            hovertemplate='Sample %{x}<br>Total Instr (min): %{y:,.0f}<extra></extra>'
        ), row=1, col=1)
        
        # Total VALU variability band
        fig.add_trace(go.Scatter(
            x=x, y=valu_max, name='Total VALU (max if all ranks had max)',
            mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
            showlegend=False,
            hovertemplate='Sample %{x}<br>Total VALU (max): %{y:,.0f}<extra></extra>'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=valu_min, name='Total VALU variability',
            mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
            fill='tonexty', fillcolor='rgba(128,128,128,0.15)',
            hovertemplate='Sample %{x}<br>Total VALU (min): %{y:,.0f}<extra></extra>'
        ), row=1, col=1)
    
    # Chart 1: Lines for totals
    fig.add_trace(go.Scatter(
        x=x, y=total_insts, name='Total Instructions',
        mode='lines', line=dict(width=2, color='black'),
        hovertemplate='Sample %{x}<br>Total Instr: %{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=valu_insts, name='Total VALU',
        mode='lines', line=dict(width=2, color='black', dash='dash'),
        hovertemplate='Sample %{x}<br>Total VALU: %{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    # Chart 2: Percentage stacked area
    for i, (col, label, color) in enumerate(zip(columns, labels, colors_list)):
        fill = 'tozeroy' if i == 0 else 'tonexty'
        fig.add_trace(go.Scatter(
            x=x, y=pct_data[col], name=f'{label} %', fill=fill,
            mode='lines', line=dict(width=0.5, color=color),
            stackgroup='pct', groupnorm='',
            hovertemplate=f'Sample %{{x}}<br>{label}: %{{y:.2f}}%<extra></extra>',
            showlegend=False
        ), row=2, col=1)
    
    # Update layout
    if num_files == 1:
        title_text = f'{set_name} Monitor Time-Series Analysis'
    else:
        title_text = f'{set_name} Monitor Time-Series Analysis ({num_files} files aggregated)'
    
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            font=dict(size=16)
        ),
        height=800,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text='Sample Period', row=1, col=1)
    fig.update_xaxes(title_text='Sample Period', row=2, col=1)
    fig.update_yaxes(title_text='Instruction Count', row=1, col=1)
    fig.update_yaxes(title_text=f'Percentage of {denom_label} (%)', range=[0, 100], row=2, col=1)
    
    # Save to HTML
    fig.write_html(output_file, include_plotlyjs=True, full_html=True)
    
    print(f"Generated: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate time-series plots from FP64 Monitor CSV output'
    )
    parser.add_argument('csv_files', nargs='+', help='Input CSV file(s) from fp64_monitor')
    parser.add_argument('--output', '-o', help='Output filename prefix (default: fp64_monitor)')
    parser.add_argument('--html', action='store_true', 
                        help='Generate interactive HTML using plotly (default: PDF using matplotlib)')
    parser.add_argument('--counters', '-c', choices=['fp64', 'fp32', 'int', 'salu', 'highlevel', 'highlevel2', 'auto'],
                        help='Counter set to plot (default: fp64, use "auto" to detect)')
    
    args = parser.parse_args()
    
    # Verify all files exist
    for csv_file in args.csv_files:
        if not Path(csv_file).exists():
            print(f"Error: File not found: {csv_file}")
            sys.exit(1)
    
    print(f"Processing {len(args.csv_files)} file(s)...")
    
    if args.html:
        result = generate_html_plots(args.csv_files, args.output, args.counters)
    else:
        result = generate_plots(args.csv_files, args.output, args.counters)
    
    if result is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
