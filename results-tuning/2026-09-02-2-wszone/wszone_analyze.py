#!/usr/bin/env python3
"""wszone sweep analysis: per arm per size, median out-of-place busbw across reps
vs the default arm; flag wins beyond both 5% tolerance and measured rep spread."""
import csv, glob, os, re, statistics as st, sys

LOGDIR = sys.argv[1] if len(sys.argv) > 1 else "."
SIZES = [67108864, 134217728, 268435456, 536870912]
SNAME = {67108864: "64M", 134217728: "128M", 268435456: "256M", 536870912: "512M"}

# ---- parse stdout logs ----
# data[arm][size] = list of (rep, oop_busbw, algo, proto)
data, rc_bad = {}, {}
for f in sorted(glob.glob(os.path.join(LOGDIR, "*_stdout.log"))):
    tag = os.path.basename(f)[:-len("_stdout.log")]
    m = re.match(r"(.+)_r(\d+)$", tag)
    arm, rep = m.group(1), int(m.group(2))
    rows = {}
    for line in open(f, errors="replace"):
        lm = re.match(r"\s+(\d+)\s+\d+\s+float\s+sum\s+-1\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
        if lm and int(lm.group(1)) in SIZES:
            size = int(lm.group(1))
            rows[size] = (float(lm.group(4)), lm.group(9), lm.group(10))  # oop busbw, algo, proto
    if len(rows) < len(SIZES):
        rc_bad.setdefault(arm, []).append(rep)
        continue
    for size, (bw, algo, proto) in rows.items():
        data.setdefault(arm, {}).setdefault(size, []).append((rep, bw, algo, proto))

# ---- verification: applied counts (skip arms w/ zero Applied) ----
verified_ok, verify_notes = {}, {}
vcsv = os.path.join(LOGDIR, "verify.csv")
applied = {}
if os.path.exists(vcsv):
    for row in csv.DictReader(open(vcsv)):
        applied[row["tag"]] = row
for arm in data:
    if arm == "default":
        verified_ok[arm] = True
        continue
    oks, zeros, mism = 0, 0, 0
    for tag, row in applied.items():
        if not re.match(re.escape(arm) + r"_r\d+$", tag):
            continue
        counts = [int(row[c]) for c in ("applied_64M", "applied_128M", "applied_256M", "applied_512M")]
        if all(c > 0 for c in counts):
            oks += 1
        else:
            zeros += 1
        mism += int(row["combo_mismatch"])
    verified_ok[arm] = oks > 0 and mism == 0
    verify_notes[arm] = f"reps applied@all4sizes={oks} zeroreps={zeros} combo_mismatch_lines={mism}"

# ---- stats ----
def med(x): return st.median(x)

print("=== per-arm, per-size: median oop busbw (GB/s), [min..max], n reps, algo/proto seen ===")
base = {}
for size in SIZES:
    bws = [b for (_, b, _, _) in data["default"][size]]
    base[size] = med(bws)
    print(f"default {SNAME[size]}: median={base[size]:.2f} [{min(bws):.2f}..{max(bws):.2f}] n={len(bws)}")

results = []  # (arm, size, medbw, delta_pct, win_flag, spread info)
for arm in sorted(data):
    if arm == "default":
        continue
    if not verified_ok.get(arm, False):
        print(f"\n{arm}: IGNORED (verification failed: {verify_notes.get(arm)})")
        continue
    print(f"\n{arm}: {verify_notes.get(arm, '')}")
    for size in SIZES:
        if size not in data[arm]:
            print(f"  {SNAME[size]}: NO DATA")
            continue
        entries = data[arm][size]
        bws = [b for (_, b, _, _) in entries]
        algos = sorted({(a, p) for (_, _, a, p) in entries})
        m = med(bws)
        d = 100.0 * (m - base[size]) / base[size]
        dbase = [b for (_, b, _, _) in data["default"][size]]
        spread = max(max(bws) - min(bws), max(dbase) - min(dbase))
        win = d > 5.0 and (m - base[size]) > spread
        loss = d < -5.0 and (base[size] - m) > spread
        flag = " ***WIN***" if win else (" (loss)" if loss else "")
        print(f"  {SNAME[size]}: median={m:.2f} [{min(bws):.2f}..{max(bws):.2f}] n={len(bws)} "
              f"delta={d:+.1f}% spread={spread:.2f} {algos}{flag}")
        results.append((arm, SNAME[size], m, d, win))

print("\n=== incomplete/crashed reps ===")
for arm, reps in sorted(rc_bad.items()):
    print(f"{arm}: reps missing size rows (crash/timeout): {reps}")

print("\n=== WINNERS per size (beyond 5% AND rep spread) ===")
for size in SIZES:
    wins = [(a, m, d) for (a, s, m, d, w) in results if s == SNAME[size] and w]
    wins.sort(key=lambda x: -x[1])
    if wins:
        for a, m, d in wins:
            print(f"{SNAME[size]}: {a} median={m:.2f} ({d:+.1f}% vs default {base[size]:.2f})")
    else:
        best = max(((a, m, d) for (a, s, m, d, w) in results if s == SNAME[size]), key=lambda x: x[1], default=None)
        print(f"{SNAME[size]}: no qualified winner (best non-qualified: {best})")
