#!/usr/bin/env python3
"""
Merge all metrics.csv files from run directories in sweep_results.

This script finds all run_* directories containing metrics.csv files
and merges them into a single combined CSV file.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path


def find_metrics_files(
    base_path: Path,
    metrics_filename: str = "metrics.csv",
    dir_suffix: str | None = None,
) -> list[tuple[str, Path]]:
    """Find all metrics files in run_* directories.
    
    Args:
        base_path: Path to sweep_results directory
        metrics_filename: Name of the metrics file to look for
        dir_suffix: Optional suffix to filter directories (e.g., '_base')
    """
    metrics_files = []
    
    for item in sorted(base_path.iterdir()):
        if item.is_dir() and item.name.startswith("run_"):
            # Filter by directory suffix if specified
            if dir_suffix and not item.name.endswith(dir_suffix):
                continue
            metrics_path = item / metrics_filename
            if metrics_path.exists():
                metrics_files.append((item.name, metrics_path))
    
    return metrics_files



# GPU-107: the median step used to live nowhere.
#
# A sweep measures each configuration REPEATS times, so merged output holds one row per repeat.
# optimize_metrics.py then picks a winner per size -- and if it is handed those raw rows it takes a
# max over every repeat of every configuration, i.e. the single luckiest sample out of ~90-160 per
# size. It also computes its tolerance window from that inflated best, widening it further. Both
# failure modes compound.
#
# The collapse to a median was previously done by python typed into a shell and never saved: only
# merged_median.csv survived, the code did not. autotune/pipeline.py still feeds optimize_metrics.py
# the RAW merged file, so the automated path has always had this bug.
#
# Stdlib only, deliberately: the dev VM has no pandas, and a step this load-bearing must run
# wherever the CSVs are.
# The requested_* columns are part of the key on purpose. A run left UNFORCED (RCCL chooses) and a
# run FORCED to those same values are different measurements even when -A 1 reports them identically
# -- 2-node 32K measured 1.510/1.490/1.510 unforced against 1.500/1.370/1.370 forced to the same
# TREE/LL/16. Collapsing them together would hide exactly the comparison the sweep exists to make:
# the unforced rows ARE the default reference every candidate is scored against.
GROUP_KEYS = ("collective", "num_nodes", "num_gpus", "size_bytes", "algo", "proto", "nchannels",
              "requested_algo", "requested_proto", "requested_nchannels")


def median_rows(rows):
    """Collapse repeats of the same (config, size) to one row of medians.

    Returns (rows, fieldnames). Numeric columns are medianed; non-numeric ones take the first value
    seen. Adds n_repeats and spread_pct, where spread is (max-min)/median of busbw_ip -- the column
    the ranking actually uses.
    """
    groups = {}
    order = []
    for r in rows:
        key = tuple(r.get(k, "") for k in GROUP_KEYS)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    out = []
    for key in order:
        members = groups[key]
        merged = dict(members[0])
        for col in members[0]:
            vals = []
            for m in members:
                try:
                    vals.append(float(m[col]))
                except (TypeError, ValueError):
                    vals = []
                    break
            if vals:
                med = statistics.median(vals)
                merged[col] = str(int(med)) if all(float(v).is_integer() for v in vals) else f"{med:g}"
        bw = []
        for m in members:
            try:
                bw.append(float(m.get("busbw_ip", "")))
            except (TypeError, ValueError):
                pass
        merged["n_repeats"] = str(len(members))
        if bw and statistics.median(bw) > 0:
            merged["spread_pct"] = f"{(max(bw) - min(bw)) / statistics.median(bw) * 100:g}"
        else:
            merged["spread_pct"] = "0"
        out.append(merged)

    fields = list(members[0].keys()) if out else []
    for extra in ("n_repeats", "spread_pct"):
        if extra not in fields:
            fields.append(extra)
    return out, fields


def merge_metrics(
    base_path: Path,
    output_file: Path,
    add_run_column: bool = False,
    metrics_filename: str = "metrics.csv",
    dir_suffix: str | None = None,
    median: bool = False,
) -> int:
    """
    Merge all metrics files from run directories.
    
    Args:
        base_path: Path to sweep_results directory
        output_file: Path to save the merged CSV
        add_run_column: Whether to add a column identifying the source run
        metrics_filename: Name of the metrics file to look for
        dir_suffix: Optional suffix to filter directories (e.g., '_base')
        
    Returns:
        Total number of data rows merged
    """
    metrics_files = find_metrics_files(base_path, metrics_filename, dir_suffix)
    
    if not metrics_files:
        print(f"No metrics.csv files found in run_* directories under {base_path}")
        sys.exit(1)
    
    print(f"Found {len(metrics_files)} metrics.csv files:")
    for run_name, path in metrics_files:
        print(f"  - {run_name}: {path}")
    
    total_rows = 0
    header_written = False
    header = None
    
    with open(output_file, 'w', newline='') as outfile:
        writer = None
        
        for run_name, metrics_path in metrics_files:
            try:
                with open(metrics_path, 'r', newline='') as infile:
                    reader = csv.reader(infile)
                    file_header = next(reader)
                    
                    # Initialize writer with first file's header
                    if not header_written:
                        if add_run_column:
                            header = ['run_id'] + file_header
                        else:
                            header = file_header
                        writer = csv.writer(outfile)
                        writer.writerow(header)
                        header_written = True
                    else:
                        # Verify headers match
                        expected_header = header[1:] if add_run_column else header
                        if file_header != expected_header:
                            print(f"  Warning: Header mismatch in {metrics_path}")
                            print(f"    Expected: {expected_header}")
                            print(f"    Got: {file_header}")
                    
                    # Write data rows
                    row_count = 0
                    for row in reader:
                        if add_run_column:
                            writer.writerow([run_name] + row)
                        else:
                            writer.writerow(row)
                        row_count += 1
                    
                    total_rows += row_count
                    print(f"  Loaded {row_count} rows from {run_name}")
                    
            except Exception as e:
                print(f"  Warning: Failed to read {metrics_path}: {e}")
    
    if total_rows == 0:
        print("No data could be loaded from any metrics.csv files")
        sys.exit(1)
    
    print(f"\nMerged {total_rows} total rows into {output_file}")

    if median:
        with open(output_file, newline="") as fh:
            rows = list(csv.DictReader(fh))
        collapsed, fields = median_rows(rows)
        with open(output_file, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(collapsed)
        print(f"Collapsed {total_rows} repeat rows -> {len(collapsed)} median rows "
              f"({total_rows / len(collapsed):.1f} repeats per configuration)")
        total_rows = len(collapsed)
    return total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Merge all metrics files from run directories"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("./sweep_results"),
        help="Path to sweep_results directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (default: <base_path>/merged_metrics.csv)",
    )
    parser.add_argument(
        "--add-run-column",
        action="store_true",
        help="Add a column to identify the source run",
    )
    parser.add_argument(
        "--metrics-filename",
        type=str,
        default="metrics.csv",
        help="Name of the metrics file to merge (default: %(default)s)",
    )
    parser.add_argument(
        "--median",
        action="store_true",
        help="collapse repeats of the same (collective, nodes, gpus, size, algo, proto, channels) "
             "to one row of medians, adding n_repeats and spread_pct. Use this before "
             "optimize_metrics.py: given raw repeat rows it selects on the single luckiest sample "
             "per size, and computes its tolerance window from that inflated best.",
    )
    parser.add_argument(
        "--dir-suffix",
        type=str,
        default=None,
        help="Only include run directories ending with this suffix (e.g., '_base')",
    )
    
    args = parser.parse_args()
    
    if not args.base_path.exists():
        print(f"Error: Base path does not exist: {args.base_path}")
        sys.exit(1)
    
    output_file = args.output or (args.base_path / "merged_metrics.csv")
    
    merge_metrics(
        base_path=args.base_path,
        output_file=output_file,
        add_run_column=args.add_run_column,
        metrics_filename=args.metrics_filename,
        dir_suffix=args.dir_suffix,
        median=args.median,
    )


if __name__ == "__main__":
    main()
