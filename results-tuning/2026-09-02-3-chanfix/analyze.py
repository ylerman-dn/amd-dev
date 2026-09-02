#!/usr/bin/env python3
"""chanfix finals analysis — pure python (no scipy on VM).

Reads bench_round*.log files under BASE/qcf_<arm>/ and produces:
  - per-arm median tok/s, mean TTFT, mean TPOT
  - MWU (normal approx, tie-corrected) of tok/s vs the 'none' arm
  - paired per-round diffs vs none (mean diff, sign counts, sign-test p)
Writes fin_stats.csv (arm,round,tok_s,mean_ttft_ms,mean_tpot_ms).
"""
import glob, math, os, re, sys, statistics

BASE = sys.argv[1] if len(sys.argv) > 1 else "."
ARMS = ["none", "current", "rs112", "rs168", "rs224"]

RX = {
    "tok_s": re.compile(r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
    "mean_ttft": re.compile(r"Mean TTFT \(ms\):\s+([0-9.]+)"),
    "mean_tpot": re.compile(r"Mean TPOT \(ms\):\s+([0-9.]+)"),
}

def load(arm):
    rows = {}
    for f in glob.glob(os.path.join(BASE, f"qcf_{arm}", "bench_round*.log")):
        r = int(re.search(r"bench_round(\d+)\.log", f).group(1))
        txt = open(f, errors="replace").read()
        vals = {}
        for k, rx in RX.items():
            m = rx.search(txt)
            if m:
                vals[k] = float(m.group(1))
        if "tok_s" in vals:
            rows[r] = vals
        else:
            print(f"WARN: {f} missing tok/s (failed rep?)", file=sys.stderr)
    return rows

def mwu(x, y):
    """Mann-Whitney U, two-sided, normal approx with tie correction."""
    n1, n2 = len(x), len(y)
    allv = [(v, 0) for v in x] + [(v, 1) for v in y]
    allv.sort()
    ranks, i = {}, 0
    N = n1 + n2
    rk = [0.0] * N
    ties = []
    while i < N:
        j = i
        while j + 1 < N and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rk[k] = r
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    R1 = sum(r for r, (v, g) in zip(rk, allv) if g == 0)
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    tie_term = sum(t**3 - t for t in ties)
    sigma2 = n1 * n2 / 12 * ((N + 1) - tie_term / (N * (N - 1)))
    if sigma2 <= 0:
        return U1, float("nan"), float("nan")
    z = (U1 - mu - math.copysign(0.5, U1 - mu)) / math.sqrt(sigma2) if U1 != mu else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return U1, z, p

def sign_test_p(pos, neg):
    """two-sided exact binomial sign test."""
    n = pos + neg
    if n == 0:
        return float("nan")
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)

data = {a: load(a) for a in ARMS}
none = data["none"]

with open(os.path.join(BASE, "fin_stats.csv"), "w") as fh:
    fh.write("arm,round,tok_s,mean_ttft_ms,mean_tpot_ms\n")
    for a in ARMS:
        for r in sorted(data[a]):
            v = data[a][r]
            fh.write(f"{a},{r},{v['tok_s']},{v.get('mean_ttft')},{v.get('mean_tpot')}\n")

print(f"{'arm':8s} {'n':>3s} {'med tok/s':>10s} {'d_vs_none':>9s} {'mean TTFT':>10s} {'mean TPOT':>10s} {'MWU z':>7s} {'p':>9s} {'paired mean d':>13s} {'signs':>8s} {'sign p':>8s}")
mn = statistics.median([v["tok_s"] for v in none.values()]) if none else float("nan")
for a in ARMS:
    d = data[a]
    if not d:
        print(f"{a:8s} MISSING")
        continue
    tok = [d[r]["tok_s"] for r in sorted(d)]
    med = statistics.median(tok)
    ttft = statistics.mean([d[r]["mean_ttft"] for r in sorted(d) if "mean_ttft" in d[r]])
    tpot = statistics.mean([d[r]["mean_tpot"] for r in sorted(d) if "mean_tpot" in d[r]])
    if a == "none":
        print(f"{a:8s} {len(tok):3d} {med:10.2f} {'—':>9s} {ttft:10.1f} {tpot:10.2f}")
        continue
    delta = (med - mn) / mn * 100
    ntok = [none[r]["tok_s"] for r in sorted(none)]
    U, z, p = mwu(tok, ntok)
    common = sorted(set(d) & set(none))
    diffs = [d[r]["tok_s"] - none[r]["tok_s"] for r in common]
    pos = sum(1 for x in diffs if x > 0)
    neg = sum(1 for x in diffs if x < 0)
    md = statistics.mean(diffs) if diffs else float("nan")
    sp = sign_test_p(pos, neg)
    print(f"{a:8s} {len(tok):3d} {med:10.2f} {delta:+8.2f}% {ttft:10.1f} {tpot:10.2f} {z:7.2f} {p:9.2e} {md:+13.2f} {pos:+3d}/-{neg:<3d} {sp:8.3f}")
