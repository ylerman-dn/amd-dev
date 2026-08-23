#!/usr/bin/env python3
"""GPU-107 parity analyzer: per-size A-vs-B across ALL sizes + selection identity + load proof.

Usage: analyze_parity.py <dir> <globA> <globB> <labelA> <labelB>
e.g.   analyze_parity.py parity_3n/logs '3n_default_r*.log' '3n_config_r*.log' v4 v5
"""
import glob
import itertools
import os
import statistics
import sys


def parse_log(path):
    """{size: (inplace_busbw, algo, proto, nch)} from one benchmark stdout."""
    out = {}
    for line in open(path, errors="ignore"):
        f = line.split()
        if len(f) >= 16 and f[0].isdigit() and "float" in f[2]:
            try:
                out[int(f[0])] = (float(f[11]), f[13], f[14], int(f[15]))
            except (ValueError, IndexError):
                continue
    return out


def psup(a, b):
    if not a or not b:
        return 0.0
    wins = sum(1 for x, y in itertools.product(a, b) if x > y)
    ties = sum(1 for x, y in itertools.product(a, b) if x == y)
    return (wins + 0.5 * ties) / (len(a) * len(b))


def main():
    d, ga, gb, la, lb = sys.argv[1:6]
    runs = {la: sorted(glob.glob(os.path.join(d, ga))), lb: sorted(glob.glob(os.path.join(d, gb)))}
    data = {la: {}, lb: {}}   # label -> size -> [(bw, algo, proto, nch)]
    for lab in (la, lb):
        if not runs[lab]:
            print(f"NO LOGS for {lab} ({d}/{ga if lab==la else gb})"); sys.exit(2)
        for p in runs[lab]:
            for size, rec in parse_log(p).items():
                data[lab].setdefault(size, []).append(rec)
    print(f"{la}: {len(runs[la])} runs   {lb}: {len(runs[lb])} runs\n")

    sizes = sorted(set(data[la]) & set(data[lb]))
    print(f"{'size':>10} {'n_'+la:>5} {'n_'+lb:>5} {la+'_med':>9} {lb+'_med':>9} {'gain%':>7} "
          f"{'psup':>5} {la+'_spr%':>7} {lb+'_spr%':>7}  selections")
    worst_gain, worst_size, sel_mismatch = 0.0, None, 0
    for s in sizes:
        va = [r[0] for r in data[la][s]]
        vb = [r[0] for r in data[lb][s]]
        ma, mb = statistics.median(va), statistics.median(vb)
        gain = (mb - ma) / ma * 100 if ma else float("nan")
        ps = psup(vb, va)
        spr = lambda v: (max(v) - min(v)) / min(v) * 100 if min(v) > 0 else float("inf")
        sela = {(r[1], r[2], r[3]) for r in data[la][s]}
        selb = {(r[1], r[2], r[3]) for r in data[lb][s]}
        selnote = "SAME" if sela == selb else f"DIFF {sorted(sela)} vs {sorted(selb)}"
        if sela != selb:
            sel_mismatch += 1
        if abs(gain) > abs(worst_gain):
            worst_gain, worst_size = gain, s
        print(f"{s:>10} {len(va):>5} {len(vb):>5} {ma:>9.2f} {mb:>9.2f} {gain:>+7.2f} "
              f"{ps:>5.2f} {spr(va):>7.1f} {spr(vb):>7.1f}  {selnote}")

    print(f"\nworst |gain|: {worst_gain:+.2f}% at {worst_size} bytes")
    print(f"selection mismatches: {sel_mismatch} of {len(sizes)} sizes")

    # load proof from the NCCL debug files
    for lab, pat in ((la, ga), (lb, gb)):
        base = pat.replace("*.log", "*_dbg_*.log").replace(".log", "_dbg_*.log") \
               if "_dbg_" not in pat else pat
        dbg = sorted(glob.glob(os.path.join(d, pat.replace(".log", "_dbg_*.log"))))
        v4n = v5n = applied = ovr = 0
        for p in dbg:
            t = open(p, errors="ignore").read()
            v4n += t.count("TUNER/Plugin: Using DN-TUNER (v4)")
            v5n += t.count("TUNER/Plugin: Using DN-TUNER (v5)")
            applied += t.count("Applied config")
            ovr += t.count("constants override")
        print(f"{lab}: {len(dbg)} dbg files | Using(v4)={v4n} Using(v5)={v5n} "
              f"applied-config lines={applied} constants-override lines={ovr}")


if __name__ == "__main__":
    main()
