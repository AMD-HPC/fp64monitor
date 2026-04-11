#!/usr/bin/env python3
"""
Time-series analysis of FP64 Monitor CSV output.

Generates charts showing instruction mix over time:
  1. Raw instruction counts per sample period
  2. Percentage of VALU per sample period

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
    python3 plot_timeseries.py <csv_files...> [--output prefix] [--html]
    python3 plot_timeseries.py fp64_monitor_*.csv --output my_analysis
    python3 plot_timeseries.py fp64_monitor_*.csv --html
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


def compute_deltas(rows):
    """Compute per-period deltas from cumulative counter values."""
    if not rows:
        return []
    
    deltas = []
    prev = None
    
    for row in rows:
        current = {
            'sample_id': int(row.get('sample_id', 0)),
            'elapsed_seconds': float(row.get('elapsed_seconds', 0)),
            'total_instructions': float(row.get('total_instructions', 0)),
            'valu_instructions': float(row.get('valu_instructions', 0)),
            'fp64_add': float(row.get('fp64_add', 0)),
            'fp64_mul': float(row.get('fp64_mul', 0)),
            'fp64_fma': float(row.get('fp64_fma', 0)),
            'fp64_trans': float(row.get('fp64_trans', 0)),
            'fp64_mfma': float(row.get('fp64_mfma', 0)),
        }
        
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
                'fp64_add': current['fp64_add'] - prev['fp64_add'],
                'fp64_mul': current['fp64_mul'] - prev['fp64_mul'],
                'fp64_fma': current['fp64_fma'] - prev['fp64_fma'],
                'fp64_trans': current['fp64_trans'] - prev['fp64_trans'],
                'fp64_mfma': current['fp64_mfma'] - prev['fp64_mfma'],
            }
        
        deltas.append(delta)
        prev = current
    
    return deltas


def load_and_aggregate(csv_files):
    """Load data from multiple CSV files and aggregate by sample period.
    
    Returns aggregated data with min/max for variability bands.
    """
    # Load deltas from each file
    all_deltas = []
    for csv_file in csv_files:
        rows = load_csv(csv_file)
        if rows:
            deltas = compute_deltas(rows)
            all_deltas.append(deltas)
    
    if not all_deltas:
        return []
    
    num_ranks = len(all_deltas)
    
    # Find the maximum number of samples across all files
    max_samples = max(len(d) for d in all_deltas)
    
    # Aggregate by sample_id (sum across all ranks, track min/max per rank)
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
            'fp64_add': 0,
            'fp64_mul': 0,
            'fp64_fma': 0,
            'fp64_trans': 0,
            'fp64_mfma': 0,
        }
        
        count = 0
        for deltas in all_deltas:
            if i < len(deltas):
                d = deltas[i]
                agg['total_instructions'] += d['total_instructions']
                agg['valu_instructions'] += d['valu_instructions']
                agg['fp64_add'] += d['fp64_add']
                agg['fp64_mul'] += d['fp64_mul']
                agg['fp64_fma'] += d['fp64_fma']
                agg['fp64_trans'] += d['fp64_trans']
                agg['fp64_mfma'] += d['fp64_mfma']
                agg['elapsed_seconds'] = max(agg['elapsed_seconds'], d['elapsed_seconds'])
                rank_total_insts.append(d['total_instructions'])
                rank_valu_insts.append(d['valu_instructions'])
                count += 1
        
        if count > 0:
            # Store min/max across ranks (scaled to total by multiplying by num_ranks)
            # This gives the range if all ranks had the min or max value
            agg['total_insts_min'] = min(rank_total_insts) * num_ranks if rank_total_insts else 0
            agg['total_insts_max'] = max(rank_total_insts) * num_ranks if rank_total_insts else 0
            agg['valu_insts_min'] = min(rank_valu_insts) * num_ranks if rank_valu_insts else 0
            agg['valu_insts_max'] = max(rank_valu_insts) * num_ranks if rank_valu_insts else 0
            aggregated.append(agg)
    
    return aggregated


def plot_raw_counts(ax, deltas):
    """Plot raw instruction counts as lines + stacked area."""
    if not deltas:
        return
    
    x = [d['sample_id'] for d in deltas]
    total_insts = [d['total_instructions'] for d in deltas]
    valu_insts = [d['valu_instructions'] for d in deltas]
    fp64_add = [d['fp64_add'] for d in deltas]
    fp64_mul = [d['fp64_mul'] for d in deltas]
    fp64_fma = [d['fp64_fma'] for d in deltas]
    fp64_trans = [d['fp64_trans'] for d in deltas]
    fp64_mfma = [d['fp64_mfma'] for d in deltas]
    
    # Check if we have variability data (from multi-rank aggregation)
    has_variability = 'total_insts_min' in deltas[0]
    
    # Stacked area for FP64 components
    ax.stackplot(x, fp64_add, fp64_mul, fp64_fma, fp64_trans, fp64_mfma,
                 labels=['FP64 ADD', 'FP64 MUL', 'FP64 FMA', 'FP64 Trans', 'FP64 MFMA'],
                 colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                 alpha=0.7)
    
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
    ax.set_title('Instruction Mix Over Time (Raw Counts)')
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis with commas for thousands
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )


def plot_percentage(ax, deltas):
    """Plot FP64 components as percentage of VALU."""
    if not deltas:
        return
    
    x = [d['sample_id'] for d in deltas]
    
    # Compute percentages (handle division by zero)
    pct_add = []
    pct_mul = []
    pct_fma = []
    pct_trans = []
    pct_mfma = []
    
    for d in deltas:
        valu = d['valu_instructions']
        if valu > 0:
            pct_add.append(100.0 * d['fp64_add'] / valu)
            pct_mul.append(100.0 * d['fp64_mul'] / valu)
            pct_fma.append(100.0 * d['fp64_fma'] / valu)
            pct_trans.append(100.0 * d['fp64_trans'] / valu)
            pct_mfma.append(100.0 * d['fp64_mfma'] / valu)
        else:
            pct_add.append(0)
            pct_mul.append(0)
            pct_fma.append(0)
            pct_trans.append(0)
            pct_mfma.append(0)
    
    # Stacked area for FP64 percentage
    ax.stackplot(x, pct_add, pct_mul, pct_fma, pct_trans, pct_mfma,
                 labels=['FP64 ADD', 'FP64 MUL', 'FP64 FMA', 'FP64 Trans', 'FP64 MFMA'],
                 colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                 alpha=0.7)
    
    ax.set_xlabel('Sample Period')
    ax.set_ylabel('Percentage of VALU (%)')
    ax.set_title('FP64 Instruction Mix Over Time (% of VALU)')
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)


def generate_plots(csv_files, output_prefix=None):
    """Generate PDF with time-series plots."""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for plotting.")
        print("Install with: pip install matplotlib")
        return None
    
    # Handle single file or list
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    
    # Load and aggregate data from all files
    if len(csv_files) == 1:
        rows = load_csv(csv_files[0])
        if not rows:
            print(f"Error: No data found in {csv_files[0]}")
            return None
        deltas = compute_deltas(rows)
    else:
        deltas = load_and_aggregate(csv_files)
    
    if not deltas:
        print("Error: Could not compute deltas from input files")
        return None
    
    # Determine output filename
    if output_prefix is None:
        output_prefix = "fp64_monitor"
    output_file = f"{output_prefix}_timeseries.pdf"
    
    # Create title
    if len(csv_files) == 1:
        title = f'FP64 Monitor Time-Series Analysis\n{Path(csv_files[0]).name}'
    else:
        title = f'FP64 Monitor Time-Series Analysis\n({len(csv_files)} files aggregated)'
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Plot raw counts (full timeline including idle periods)
    plot_raw_counts(ax1, deltas)
    
    # Plot percentages
    plot_percentage(ax2, deltas)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save to PDF
    plt.savefig(output_file, format='pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated: {output_file}")
    return output_file


def generate_html_plots(csv_files, output_prefix=None):
    """Generate interactive HTML with time-series plots using plotly."""
    if not HAS_PLOTLY:
        print("Error: plotly is required for HTML output.")
        print("Install with: pip install plotly")
        return None
    
    # Handle single file or list
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    
    # Load and aggregate data from all files
    if len(csv_files) == 1:
        rows = load_csv(csv_files[0])
        if not rows:
            print(f"Error: No data found in {csv_files[0]}")
            return None
        deltas = compute_deltas(rows)
    else:
        deltas = load_and_aggregate(csv_files)
    
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
    fp64_add = [d['fp64_add'] for d in deltas]
    fp64_mul = [d['fp64_mul'] for d in deltas]
    fp64_fma = [d['fp64_fma'] for d in deltas]
    fp64_trans = [d['fp64_trans'] for d in deltas]
    fp64_mfma = [d['fp64_mfma'] for d in deltas]
    
    # Check if we have variability data (from multi-rank aggregation)
    has_variability = 'total_insts_min' in deltas[0]
    if has_variability:
        total_min = [d['total_insts_min'] for d in deltas]
        total_max = [d['total_insts_max'] for d in deltas]
        valu_min = [d['valu_insts_min'] for d in deltas]
        valu_max = [d['valu_insts_max'] for d in deltas]
    
    # Compute percentages
    pct_add, pct_mul, pct_fma, pct_trans, pct_mfma = [], [], [], [], []
    for d in deltas:
        valu = d['valu_instructions']
        if valu > 0:
            pct_add.append(100.0 * d['fp64_add'] / valu)
            pct_mul.append(100.0 * d['fp64_mul'] / valu)
            pct_fma.append(100.0 * d['fp64_fma'] / valu)
            pct_trans.append(100.0 * d['fp64_trans'] / valu)
            pct_mfma.append(100.0 * d['fp64_mfma'] / valu)
        else:
            pct_add.append(0)
            pct_mul.append(0)
            pct_fma.append(0)
            pct_trans.append(0)
            pct_mfma.append(0)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Instruction Mix Over Time (Raw Counts)',
            'FP64 Instruction Mix Over Time (% of VALU)'
        ),
        vertical_spacing=0.12
    )
    
    # Colors for FP64 components
    colors = {
        'add': '#1f77b4',
        'mul': '#ff7f0e', 
        'fma': '#2ca02c',
        'trans': '#d62728',
        'mfma': '#9467bd'
    }
    
    # Chart 1: Raw counts - stacked area for FP64 components
    fig.add_trace(go.Scatter(
        x=x, y=fp64_add, name='FP64 ADD', fill='tozeroy',
        mode='lines', line=dict(width=0.5, color=colors['add']),
        stackgroup='fp64', groupnorm='',
        hovertemplate='Sample %{x}<br>FP64 ADD: %{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=fp64_mul, name='FP64 MUL', fill='tonexty',
        mode='lines', line=dict(width=0.5, color=colors['mul']),
        stackgroup='fp64',
        hovertemplate='Sample %{x}<br>FP64 MUL: %{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=fp64_fma, name='FP64 FMA', fill='tonexty',
        mode='lines', line=dict(width=0.5, color=colors['fma']),
        stackgroup='fp64',
        hovertemplate='Sample %{x}<br>FP64 FMA: %{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=fp64_trans, name='FP64 Trans', fill='tonexty',
        mode='lines', line=dict(width=0.5, color=colors['trans']),
        stackgroup='fp64',
        hovertemplate='Sample %{x}<br>FP64 Trans: %{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=fp64_mfma, name='FP64 MFMA', fill='tonexty',
        mode='lines', line=dict(width=0.5, color=colors['mfma']),
        stackgroup='fp64',
        hovertemplate='Sample %{x}<br>FP64 MFMA: %{y:,.0f}<extra></extra>'
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
    fig.add_trace(go.Scatter(
        x=x, y=pct_add, name='FP64 ADD %', fill='tozeroy',
        mode='lines', line=dict(width=0.5, color=colors['add']),
        stackgroup='pct', groupnorm='',
        hovertemplate='Sample %{x}<br>FP64 ADD: %{y:.2f}%<extra></extra>',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=pct_mul, name='FP64 MUL %', fill='tonexty',
        mode='lines', line=dict(width=0.5, color=colors['mul']),
        stackgroup='pct',
        hovertemplate='Sample %{x}<br>FP64 MUL: %{y:.2f}%<extra></extra>',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=pct_fma, name='FP64 FMA %', fill='tonexty',
        mode='lines', line=dict(width=0.5, color=colors['fma']),
        stackgroup='pct',
        hovertemplate='Sample %{x}<br>FP64 FMA: %{y:.2f}%<extra></extra>',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=pct_trans, name='FP64 Trans %', fill='tonexty',
        mode='lines', line=dict(width=0.5, color=colors['trans']),
        stackgroup='pct',
        hovertemplate='Sample %{x}<br>FP64 Trans: %{y:.2f}%<extra></extra>',
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=pct_mfma, name='FP64 MFMA %', fill='tonexty',
        mode='lines', line=dict(width=0.5, color=colors['mfma']),
        stackgroup='pct',
        hovertemplate='Sample %{x}<br>FP64 MFMA: %{y:.2f}%<extra></extra>',
        showlegend=False
    ), row=2, col=1)
    
    # Update layout
    if num_files == 1:
        title_text = 'FP64 Monitor Time-Series Analysis'
    else:
        title_text = f'FP64 Monitor Time-Series Analysis ({num_files} files aggregated)'
    
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
    fig.update_yaxes(title_text='Percentage of VALU (%)', range=[0, 100], row=2, col=1)
    
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
    
    args = parser.parse_args()
    
    # Verify all files exist
    for csv_file in args.csv_files:
        if not Path(csv_file).exists():
            print(f"Error: File not found: {csv_file}")
            sys.exit(1)
    
    print(f"Processing {len(args.csv_files)} file(s)...")
    
    if args.html:
        result = generate_html_plots(args.csv_files, args.output)
    else:
        result = generate_plots(args.csv_files, args.output)
    
    if result is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
