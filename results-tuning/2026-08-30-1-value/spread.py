#!/usr/bin/env python3
"""Per-repeat busbw at one size, per arm — to see the repeat spread behind a median.

Usage: spread.py <dir> <size-bytes> <arm>...
"""
import csv
import glob
import os
import sys

I_SIZE, I_INPLACE, I_BUSBW = 6, 9, 12


def main():
    directory, size, arms = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    for arm in arms:
        vals = []
        for path in sorted(glob.glob(os.path.join(directory, f"{arm}_rep*.csv"))):
            with open(path, newline="") as fh:
                for row in csv.reader(fh):
                    if len(row) < 14 or row[I_INPLACE].strip('"') != "0":
                        continue
                    try:
                        if int(row[I_SIZE]) == size:
                            vals.append(float(row[I_BUSBW]))
                    except ValueError:
                        continue
        if vals:
            spread = (max(vals) - min(vals)) / min(vals) * 100
            print(f"{arm:>10}  " + " ".join(f"{v:8.2f}" for v in vals) +
                  f"   spread={spread:5.1f}%")


if __name__ == "__main__":
    main()
