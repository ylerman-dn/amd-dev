#!/usr/bin/env python3
"""Test the NIC-exporter hypothesis against the 2026-08-25 5-node preflight refusals.

Hypothesis under test: the per-node metrics exporters (30 s scrape cadence,
10-11 s busy per scrape) disturb multi-node benchmarks enough to cause the
>25% same-config spreads that refused every 5n A/B attempt.

Method (offline, uses only the raw preflight/warmup logs already on disk):
  - every benchmark launch left its stdout in <attempt>/5n_*.log with the
    file mtime = run end time;
  - a (run, size) sample is an OUTLIER when its busbw_ip is more than 25%
    below the median of the same size across the runs of the same attempt
    phase (the same definition of "noisy" the preflight uses, one-sided:
    disturbances slow things down, they do not speed them up);
  - if the exporters cause the outliers and their timers are wall-aligned
    (systemd timers fire at fixed offsets), outlier run END TIMES cluster at
    fixed phases of the 30 s grid; if the timers are NOT wall-aligned the
    per-attempt inter-arrival pattern still cannot be uniform, because a
    30 s cadence sampled by ~14 s runs aliases.

This is a screening test: a flat phase histogram does NOT fully acquit the
exporters (unaligned timers with per-node drift would also look flat); a
peaked one convicts them. Run: python3 exporter_correlation.py <logs_root>
"""
import collections
import math
import os
import re
import statistics
import sys

ROW = re.compile(r"^\s+(\d+)\s+\d+\s+\w+\s+\S+\s+-?\d+\s+"
                 r"([\d.]+)\s+[\d.]+\s+[\d.]+\s+\S+\s+"
                 r"([\d.]+)\s+[\d.]+\s+([\d.]+)\s+\S+")


def parse_log(path):
    """{size: (busbw_ip, time_oop_us, time_ip_us)} from a benchmark stdout log."""
    out = {}
    for line in open(path, errors="ignore"):
        m = ROW.match(line)
        if m:
            size = int(m.group(1))
            t_oop = float(m.group(2))
            t_ip = float(m.group(3))
            bw_ip = float(m.group(4))
            out[size] = (bw_ip, t_oop, t_ip)
    return out


def main(root):
    # group logs by (attempt dir, phase): phases are preflight_r*, default_r*, config_r*
    groups = collections.defaultdict(list)   # (dir, phase) -> [(runfile, mtime, {size:...})]
    n_logs = 0
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            m = re.match(r"5n_(preflight|default|config)_r(\d+)\.log$", f)
            if not m:
                continue
            p = os.path.join(dirpath, f)
            data = parse_log(p)
            if not data:
                continue
            n_logs += 1
            groups[(os.path.basename(dirpath), m.group(1))].append(
                (f, os.path.getmtime(p), data))

    outlier_times, clean_times = [], []
    outlier_rows = []
    for (att, phase), runs in sorted(groups.items()):
        if len(runs) < 3:
            continue
        sizes = set.intersection(*(set(d) for _, _, d in runs))
        for size in sorted(sizes):
            vals = [d[size][0] for _, _, d in runs]
            med = statistics.median(vals)
            if med <= 0:
                continue
            for (f, mt, d) in runs:
                bw = d[size][0]
                # reconstruct roughly when this size executed inside the run:
                # sizes run in ascending order; each size costs about
                # (warmup 5 + iters 20) * (t_oop + t_ip). The absolute scale
                # cancels in the mod-30 test only if the model total is close
                # to the real wall time, so this stays an approximation.
                offset = sum((5 + 20) * (dd[1] + dd[2]) * 1e-6
                             for s2, dd in sorted(d.items()) if s2 < size)
                total = sum((5 + 20) * (dd[1] + dd[2]) * 1e-6
                            for dd in d.values())
                t = mt - total + offset
                if bw < med * 0.75:
                    outlier_times.append(t)
                    outlier_rows.append((att, phase, f, size,
                                         round(bw, 2), round(med, 2)))
                else:
                    clean_times.append(t)

    print(f"logs parsed: {n_logs}; outlier samples: {len(outlier_times)}; "
          f"clean samples: {len(clean_times)}")
    print("\noutliers (attempt, phase, run, size, busbw, size-median):")
    for r in outlier_rows:
        print("  ", r)

    def phase_hist(times, mod):
        h = [0] * 6
        for t in times:
            h[int((t % mod) / (mod / 6))] += 1
        return h

    def rayleigh(times, mod):
        """Resultant length of the phase vector; 1 = perfectly clustered,
        ~1/sqrt(n) expected under uniformity."""
        if not times:
            return 0.0
        x = sum(math.cos(2 * math.pi * (t % mod) / mod) for t in times)
        y = sum(math.sin(2 * math.pi * (t % mod) / mod) for t in times)
        return math.hypot(x, y) / len(times)

    for mod in (30.0, 60.0):
        ho = phase_hist(outlier_times, mod)
        hc = phase_hist(clean_times, mod)
        ro = rayleigh(outlier_times, mod)
        rc = rayleigh(clean_times, mod)
        n = len(outlier_times)
        # under uniformity R > sqrt(-ln(alpha)/n); alpha=0.01
        thresh = math.sqrt(-math.log(0.01) / n) if n else float("inf")
        print(f"\nmod {mod:.0f}s: outlier phase hist {ho} R={ro:.3f} "
              f"(uniformity reject threshold ~{thresh:.3f}); "
              f"clean hist {hc} R={rc:.3f}")
        if n:
            verdict = ("CLUSTERED - consistent with a wall-aligned periodic "
                       "disturbance" if ro > thresh else
                       "no wall-aligned clustering")
            print(f"  -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
