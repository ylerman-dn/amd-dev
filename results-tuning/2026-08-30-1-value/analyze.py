#!/usr/bin/env python3
"""Median busbw per (arm, size) from rccl-tests CSV files. Stdlib only (no pandas on the dev VM).

Usage: analyze.py <dir> <arm-prefix>...      e.g. analyze.py ./alltoall-2n def p2p8 p2p16
Reads <dir>/<arm>_rep*.csv, takes the median busbw across repeats per size,
and prints one row per size with each arm's median and its percent difference
from the first arm listed (the reference).
"""
import csv
import glob
import os
import statistics
import sys


# rccl-tests writes a 13-name header over 14-field rows, so DictReader misaligns.
# Parse positionally instead: ... size type redop inplace time algbw busbw #wrong
I_SIZE, I_INPLACE, I_BUSBW = 6, 9, 12


def medians(directory, arm):
    """size -> median busbw over all repeats, out-of-place rows only."""
    by_size = {}
    for path in sorted(glob.glob(os.path.join(directory, f"{arm}_rep*.csv"))):
        with open(path, newline="") as fh:
            for row in csv.reader(fh):
                if len(row) < 14 or row[I_INPLACE].strip('"') != "0":
                    continue
                try:
                    size = int(row[I_SIZE])
                    busbw = float(row[I_BUSBW])
                except ValueError:
                    continue
                by_size.setdefault(size, []).append(busbw)
    return {s: statistics.median(v) for s, v in by_size.items()}, {s: len(v) for s, v in by_size.items()}


def human(size):
    for unit, div in (("M", 1 << 20), ("K", 1 << 10)):
        if size >= div:
            return f"{size // div}{unit}"
    return str(size)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    directory, arms = sys.argv[1], sys.argv[2:]
    data = {arm: medians(directory, arm) for arm in arms}
    ref = arms[0]

    sizes = sorted({s for arm in arms for s in data[arm][0]})
    header = f"{'size':>6} {'n':>3} " + " ".join(f"{arm:>12}" for arm in arms) + \
             " " + " ".join(f"{arm + '%':>9}" for arm in arms[1:])
    print(header)
    print("-" * len(header))
    for size in sizes:
        cells, deltas = [], []
        base = data[ref][0].get(size)
        for arm in arms:
            val = data[arm][0].get(size)
            cells.append(f"{val:12.2f}" if val is not None else f"{'-':>12}")
            if arm != ref:
                if val is not None and base:
                    deltas.append(f"{(val - base) / base * 100:+8.2f}%")
                else:
                    deltas.append(f"{'-':>9}")
        reps = data[ref][1].get(size, 0)
        print(f"{human(size):>6} {reps:>3} " + " ".join(cells) + " " + " ".join(deltas))


if __name__ == "__main__":
    main()
