#!/usr/bin/env python3
"""Analyze gp8kv paired finals CSV: arm,round,tps,mean_ttft,median_ttft,mean_tpot.

Per-arm medians, MWU vs none (tok/s, mean TTFT), paired per-round diffs vs none.
Pure stdlib (no numpy/scipy on this VM). MWU: normal approximation, two-sided,
tie-corrected (same as results-tuning/2026-09-01-2-prefill8k/mwu.py).
"""
import sys, math, csv
from statistics import median, mean


def mwu(a, b):
    n1, n2 = len(a), len(b)
    allv = [(v, 0) for v in a] + [(v, 1) for v in b]
    allv.sort()
    ranks = [0.0] * len(allv)
    i = 0
    tie_term = 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[k] = avg
        t = j - i
        tie_term += t ** 3 - t
        i = j
    r1 = sum(r for r, (_, g) in zip(ranks, allv) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma2 <= 0:
        return u1, 1.0
    z = (u1 - mu - (0.5 if u1 > mu else -0.5 if u1 < mu else 0.0)) / math.sqrt(sigma2)
    p = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
    return u1, min(p, 1.0)


def main(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["arm"], {})[int(r["round"])] = (
                float(r["tps"]), float(r["mean_ttft"]),
                float(r["median_ttft"]), float(r["mean_tpot"]))
    arms = sorted(rows, key=lambda a: (a != "none", a))
    none = rows["none"]

    print(f"{'arm':<16}{'n':>3}{'med tok/s':>11}{'med mTTFT':>11}{'med mTPOT':>11}"
          f"{'d tok/s':>9}{'p tok/s':>10}{'d TTFT':>9}{'p TTFT':>10}")
    for a in arms:
        d = rows[a]
        tps = [v[0] for v in d.values()]
        ttft = [v[1] for v in d.values()]
        tpot = [v[3] for v in d.values()]
        line = (f"{a:<16}{len(d):>3}{median(tps):>11.2f}{median(ttft):>11.1f}"
                f"{median(tpot):>11.2f}")
        if a != "none":
            ntps = [v[0] for v in none.values()]
            nttft = [v[1] for v in none.values()]
            dtps = (median(tps) - median(ntps)) / median(ntps) * 100
            dtt = (median(ttft) - median(nttft)) / median(nttft) * 100
            _, p1 = mwu(tps, ntps)
            _, p2 = mwu(ttft, nttft)
            line += f"{dtps:>+9.2f}{p1:>10.3g}{dtt:>+9.2f}{p2:>10.3g}"
        print(line)

    print("\nPaired per-round diffs vs none (arm minus none):")
    for a in arms:
        if a == "none":
            continue
        common = sorted(set(rows[a]) & set(none))
        dt = [rows[a][r][0] - none[r][0] for r in common]
        df = [rows[a][r][1] - none[r][1] for r in common]
        print(f"  {a:<16} tok/s: mean {mean(dt):+8.2f} "
              f"({mean(dt)/mean(none[r][0] for r in common)*100:+.2f}%), "
              f"signs +{sum(1 for x in dt if x > 0)}/-{sum(1 for x in dt if x < 0)} | "
              f"mTTFT: mean {mean(df):+8.1f} ms, "
              f"signs +{sum(1 for x in df if x > 0)}/-{sum(1 for x in df if x < 0)}")


if __name__ == "__main__":
    main(sys.argv[1])
