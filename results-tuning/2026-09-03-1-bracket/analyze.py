#!/usr/bin/env python3
"""Bracket sweep analysis: per size, each arm's channel value vs the no-plugin default.

Input: <dir> with <arm>_r<rep>_stdout.log (rccl-tests -A output).
Per (arm, size): out-of-place busbw of the 5 reps. Per size: median gain vs the
default arm and P(sup) = fraction of pairwise rep wins (the validate_tuner_config.py
gate, >=0.95), plus Mann-Whitney U p-value. Also prints the -A algo/proto seen, to
verify the arm's rule was actually selected.
"""
import sys, re, glob, os, statistics as st
from itertools import product

DIR = sys.argv[1] if len(sys.argv) > 1 else '.'
ARMS = ['default'] + [f'arm{i}' for i in range(1, 6)] + [f'big{i}' for i in range(1, 6)]
SIZES = [4096*2**i for i in range(18)]  # 4K..512M

def sname(b):
    return f'{b//1048576}M' if b >= 1048576 else f'{b//1024}K'

# channel value each arm requested per size (mirrors the sw_bk_*.conf files)
CENTERS = {4096:1,8192:1,16384:1,32768:2,65536:4,131072:8,262144:16,524288:16,
           1048576:32,2097152:52,4194304:52,8388608:54,16777216:56,33554432:56}
MULTS = [0.5,0.75,1.0,1.25,1.5]
LAD112 = [56,84,112,140,168]; LAD224 = [192,208,224,240,252]
def req(arm, b):
    if arm == 'default': return 112
    k = int(arm[-1]) - 1
    if arm.startswith('big'):
        return LAD224[k] if b >= 67108864 else 112
    return LAD112[k] if b >= 67108864 else max(1, int(CENTERS[b]*MULTS[k]+0.5))

data = {}   # (arm,size) -> [busbw]
combo = {}  # (arm,size) -> set of ALGO/PROTO seen
for arm in ARMS:
    for f in sorted(glob.glob(os.path.join(DIR, f'{arm}_r*_stdout.log'))):
        for ln in open(f, errors='replace'):
            m = re.match(r'\s+(\d+)\s+\d+\s+float\s+sum\s+-1\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+(\S+)\s+(\S+)\s+(-?\d+)', ln)
            if m and int(m.group(1)) in SIZES:
                b = int(m.group(1))
                data.setdefault((arm, b), []).append(float(m.group(2)))
                combo.setdefault((arm, b), set()).add(f'{m.group(3)}/{m.group(4)}')

def mwu_p(a, b):
    # two-sided Mann-Whitney U, normal approx with tie correction
    import math
    n1, n2 = len(a), len(b)
    allv = sorted((v, i) for i, v in enumerate(a + b))
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]: j += 1
        r = (i + j + 1) / 2
        for k in range(i, j): ranks[allv[k][1]] = r
        i = j
    R1 = sum(ranks[i] for i in range(n1))
    U = R1 - n1*(n1+1)/2
    mu = n1*n2/2
    ties = {}
    for v, _ in allv: ties[v] = ties.get(v, 0) + 1
    tc = sum(t**3 - t for t in ties.values())
    n = n1 + n2
    var = n1*n2/12 * ((n+1) - tc/(n*(n-1)))
    if var == 0: return 1.0
    z = (U - mu) / math.sqrt(var)
    return math.erfc(abs(z)/math.sqrt(2))

print(f'{"size":>6} {"arm":>8} {"req_ch":>6} {"combo":>12} {"n":>2} {"median":>8} {"gain%":>7} {"P(sup)":>6} {"p":>7}')
for b in SIZES:
    dref = data.get(('default', b), [])
    if not dref: continue
    dmed = st.median(dref)
    print(f'{sname(b):>6} {"default":>8} {112:>6} {"/".join(sorted(combo.get(("default",b),{"?"}))):>12} {len(dref):>2} {dmed:>8.2f}')
    for arm in ARMS[1:]:
        v = data.get((arm, b), [])
        if not v: continue
        med = st.median(v)
        wins = sum(x > y for x, y in product(v, dref))
        psup = wins / (len(v)*len(dref))
        p = mwu_p(v, dref)
        gate = ' *' if (psup >= 0.95 and med > dmed) else ('  !' if (1-psup) >= 0.95 and med < dmed else '')
        print(f'{"":>6} {arm:>8} {req(arm,b):>6} {"/".join(sorted(combo.get((arm,b),{"?"}))):>12} {len(v):>2} {med:>8.2f} {100*(med/dmed-1):>+7.2f} {psup:>6.2f} {p:>7.3f}{gate}')
    print()
