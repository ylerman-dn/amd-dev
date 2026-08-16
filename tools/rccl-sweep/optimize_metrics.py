#!/usr/bin/env python3
"""
Optimize metrics CSV by keeping only the best performing configuration
for each unique combination of collective-type, num_nodes, num_gpus, and size_bytes.

Best is defined as the configuration with the minimum nchannels among those
that are within a tolerance percentage of the best (maximum) busbw_ip.

GPU-107: ranking is on busbw_ip, not time_ip_us -- see the comment in
optimize_metrics() for why time_ip_us is unsafe under -C/--report_cputime.
"""

import argparse
import sys
import pandas as pd
from pathlib import Path


def optimize_metrics(input_file: str, output_file: str = None, tolerance_pct: float = 5.0) -> pd.DataFrame:
    """
    Read metrics CSV and keep only the best configuration for each unique combination.
    
    Strategy: For each group, find configs within tolerance_pct of the best time_ip_us,
    then pick the one with the minimum nchannels.
    
    Args:
        input_file: Path to the input metrics.csv file
        output_file: Path to the output file. If None, uses 'metrics_optimize.csv' 
                     in the same directory as input_file
        tolerance_pct: Percentage tolerance below the best busbw_ip (default: 5%).
            NOTE: this must be LARGER than the measurement's run-to-run spread or
            the tie-break will select on noise. Check spread before trusting it.
    
    Returns:
        DataFrame with optimized metrics
    """
    # Read the input CSV
    df = pd.read_csv(input_file)

    # GPU-107: refuse to select silently on un-medianed data.
    # If the caller hands us raw sweep output, each configuration appears once per repeat. The
    # selection below takes a max over the group, so it would pick the single luckiest sample out of
    # ~90-160 per size, and then derive the tolerance window from that inflated best. Both errors
    # push in the same direction and neither is visible in the output.
    # merge_metrics.py --median is the fix; this only warns, because an operator may legitimately be
    # inspecting raw data.
    _dup_cols = [c for c in ('size_bytes', 'algo', 'proto', 'nchannels',
                             'requested_algo', 'requested_proto') if c in df.columns]
    if _dup_cols:
        _dups = int(df.duplicated(subset=_dup_cols).sum())
        if _dups:
            print(f"WARNING: {_dups} of {len(df)} rows repeat a (size, algo, proto, channels) key, "
                  f"so this looks like raw per-repeat output rather than medians.\n"
                  f"         Selection takes a max within each size, so it will pick the luckiest "
                  f"single repeat and widen the tolerance window from it.\n"
                  f"         Run: merge_metrics.py --base-path <dir> -o <file> --median",
                  file=sys.stderr)

    # Define the grouping columns
    group_cols = ['collective', 'num_nodes', 'num_gpus', 'size_bytes']

    # Verify required columns exist
    required_cols = group_cols + ['time_ip_us', 'nchannels']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # GPU-107: drop rows where RCCL did not honour the request. Their algo/proto
    # label describes a configuration that never ran, so a winner picked from them
    # would be written into the tuner config as a rule that can never fire.
    # This is exactly how the previous config ended up 37% unfireable.
    if 'substituted' in df.columns:
        n_before = len(df)
        df = df[df['substituted'] != 1].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"Dropped {n_dropped} of {n_before} rows where the request was "
                  f"silently substituted (not honoured by RCCL)")
        if df.empty:
            raise ValueError(
                "Every row was substituted - no honoured measurements to optimize. "
                "Check that the requested algo/proto combinations are supported at "
                "this node count.")
    else:
        print("WARNING: no 'substituted' column - this CSV predates the "
              "requested-vs-measured fix, so its algo/proto labels are unverified.")

    # Collect best indices for each group
    best_indices = []
    
    # GPU-107: rank on busbw_ip (higher is better), NOT time_ip_us.
    #
    # `time_ip_us` is unusable as a ranking metric whenever rccl-tests is run with
    # -C/--report_cputime: that column then reports CPU launch time, which is
    # ~constant (5.6-6.9us at 16M) regardless of how long the collective actually
    # takes. Ranking on it is ranking on noise, and the min-channels tie-break then
    # selects 1-channel configs that are ~99% slower than default. Evidence:
    # results-tuning/2026-08-04-1-sweep1node/superseded-n5/ (busbw 2.85 vs 262 GB/s
    # at 16M, both reporting ~5.7us).
    #
    # busbw is derived from the wall-clock measurement and matches our independent
    # srun baselines, so it is the safe metric.
    rank_col = 'busbw_ip' if 'busbw_ip' in df.columns else None
    if rank_col is None:
        raise ValueError("busbw_ip column required for ranking")

    for _, group in df.groupby(group_cols):
        # Best = highest bandwidth in this group
        best_bw = group[rank_col].max()
        # Configs within tolerance of the best are treated as equivalent...
        threshold = best_bw * (1 - tolerance_pct / 100)
        within_tolerance = group[group[rank_col] >= threshold]

        # ...and among equivalents prefer the FEWEST channels, since each channel
        # costs a block of GPU compute units that the model could otherwise use.
        best_idx = within_tolerance['nchannels'].astype(float).idxmin()
        best_indices.append(best_idx)
    
    # Select the best rows
    df_optimized = df.loc[best_indices].reset_index(drop=True)
    
    # Sort by the grouping columns for consistent output
    df_optimized = df_optimized.sort_values(group_cols).reset_index(drop=True)
    
    # Determine output file path
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / 'metrics_optimize.csv'
    
    # Write to output CSV
    df_optimized.to_csv(output_file, index=False)
    
    # Print summary
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Tolerance: {tolerance_pct}% below best busbw_ip")
    print(f"Original rows: {len(df)}")
    print(f"Optimized rows: {len(df_optimized)}")
    print(f"Unique combinations: {len(df_optimized)}")
    print(f"Rows removed: {len(df) - len(df_optimized)}")
    
    return df_optimized


def main():
    parser = argparse.ArgumentParser(
        description='Optimize metrics CSV by selecting min nchannels within tolerance of best time_ip_us'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input metrics.csv file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to the output file (default: metrics_optimize.csv in same directory as input)'
    )
    parser.add_argument(
        '-t', '--tolerance',
        type=float,
        default=5.0,
        help='Percentage tolerance below best busbw_ip (default: 5%%)'
    )
    
    args = parser.parse_args()
    
    optimize_metrics(args.input_file, args.output, args.tolerance)


if __name__ == '__main__':
    main()
