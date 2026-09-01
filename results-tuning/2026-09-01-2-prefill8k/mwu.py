#!/usr/bin/env python3
"""Mann-Whitney U (normal approximation, two-sided, tie-corrected) for two samples.

Usage: mwu.py fileA fileB   (one number per line)
"""
import sys, math
from statistics import median


def mwu(a, b):
    n1, n2 = len(a), len(b)
    allv = [(v, 0) for v in a] + [(v, 1) for v in b]
    allv.sort()
    # ranks with ties -> average rank
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


if __name__ == "__main__":
    a = [float(x) for x in open(sys.argv[1]) if x.strip()]
    b = [float(x) for x in open(sys.argv[2]) if x.strip()]
    u, p = mwu(a, b)
    print(f"nA={len(a)} nB={len(b)} medianA={median(a):.2f} medianB={median(b):.2f} "
          f"delta={(median(a)-median(b))/median(b)*100:+.2f}% U={u:.1f} p={p:.4g}")
