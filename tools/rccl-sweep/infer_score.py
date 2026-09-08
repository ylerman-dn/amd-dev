#!/usr/bin/env python3
"""Score an in-model A/B campaign driven by infer_paired.sh (GPU-107, 2026-09-08).

Input : the campaign base dir (IM_BASE) and the run prefix; every arm is a folder
        <base>/<prefix>_<arm>/ holding bench_round<r>.log files written by
        sglang.bench_serving, one per round. Arms were interleaved per round by the
        driver, so round r of every arm sampled the same minutes.
Output: one table (stdout + <base>/<prefix>_score.csv) per arm:
          n rounds used, median tok/s, median TTFT, median TPOT,
          gain% vs the reference arm (median vs median),
          P(sup) vs reference = fraction of same-round pairs the arm's tok/s wins
          (0.5 = coin flip, 1.0 = always faster; rank statistic, no outlier fit).
Rules: the first round of every arm is dropped (server warm-up, see README
        "In-model mode": 20-50% low). Nothing else is trimmed; medians absorb the rest.
        Rule-hit counts, when a detect run exists (<prefix>can_<arm>/ or the paired
        server's rccl logs), are reported as-is from grep, not interpreted.

Usage:
  infer_score.py --base /data/ylerman/<campaign> --prefix td_q --ref none
"""
import argparse
import csv
import glob
import os
import re
import statistics
import sys

TOKS = re.compile(r"Output token throughput \(tok/s\):\s+([0-9.]+)")
TTFT = re.compile(r"Mean TTFT \(ms\):\s+([0-9.]+)")
TPOT = re.compile(r"Mean TPOT \(ms\):\s+([0-9.]+)")
ROUND = re.compile(r"bench_round(\d+)\.log$")


def read_arm(folder):
    """{round: (tok/s, ttft_ms, tpot_ms)} for every complete bench log."""
    out = {}
    for f in glob.glob(os.path.join(folder, "bench_round*.log")):
        r = int(ROUND.search(f).group(1))
        txt = open(f, errors="ignore").read()
        m = TOKS.search(txt)
        if not m:
            continue  # incomplete/failed round: no number, no row
        t1 = TTFT.search(txt)
        t2 = TPOT.search(txt)
        out[r] = (float(m.group(1)),
                  float(t1.group(1)) if t1 else float("nan"),
                  float(t2.group(1)) if t2 else float("nan"))
    return out


def psup(a, b):
    """P(sup) of a over b on same-round pairs."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan"), 0
    wins = sum(1 for k in keys if a[k][0] > b[k][0])
    ties = sum(1 for k in keys if a[k][0] == b[k][0])
    return (wins + 0.5 * ties) / len(keys), len(keys)


def rule_hits(base, prefix, arm):
    """Count of plugin 'Applied config' lines in the detect run's server logs, if any."""
    pats = [os.path.join(base, f"{prefix}can_srv", "logs", "*.log*"),
            os.path.join(base, f"{prefix}_{arm}can_srv", "logs", "*.log*")]
    n = 0
    found = False
    for pat in pats:
        for f in glob.glob(pat):
            found = True
            opener = open
            if f.endswith(".gz"):
                import gzip
                opener = gzip.open
            with opener(f, "rt", errors="ignore") as fh:
                for line in fh:
                    if "Applied config" in line:
                        n += 1
    return n if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="IM_BASE of the campaign")
    ap.add_argument("--prefix", required=True, help="run prefix given to infer_paired.sh")
    ap.add_argument("--ref", default="none", help="reference arm name (default: none)")
    ap.add_argument("--drop-first", type=int, default=1,
                    help="rounds dropped per arm as warm-up (default 1)")
    args = ap.parse_args()

    arms = {}
    for d in sorted(glob.glob(os.path.join(args.base, f"{args.prefix}_*"))):
        name = os.path.basename(d)[len(args.prefix) + 1:]
        if name.endswith("srv") or not os.path.isdir(d):
            continue
        data = read_arm(d)
        for r in sorted(data)[:args.drop_first]:
            del data[r]
        arms[name] = data
    if args.ref not in arms:
        sys.exit(f"reference arm '{args.ref}' not found; arms: {sorted(arms)}")
    ref = arms[args.ref]
    ref_med = statistics.median(v[0] for v in ref.values()) if ref else float("nan")

    rows = []
    for name, data in arms.items():
        if not data:
            rows.append(dict(arm=name, n=0)); continue
        med = statistics.median(v[0] for v in data.values())
        p, npairs = psup(data, ref)
        rows.append(dict(
            arm=name, n=len(data),
            median_toks=round(med, 2),
            median_ttft_ms=round(statistics.median(v[1] for v in data.values()), 1),
            median_tpot_ms=round(statistics.median(v[2] for v in data.values()), 2),
            gain_pct_vs_ref=round((med / ref_med - 1) * 100, 2) if ref_med else float("nan"),
            psup_vs_ref=round(p, 2), pairs=npairs,
            rule_hits_detect=rule_hits(args.base, args.prefix, name)))
    cols = ["arm", "n", "median_toks", "median_ttft_ms", "median_tpot_ms",
            "gain_pct_vs_ref", "psup_vs_ref", "pairs", "rule_hits_detect"]
    out = os.path.join(args.base, f"{args.prefix}_score.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"# ref={args.ref} first {args.drop_first} round(s) per arm dropped as warm-up")
    print(",".join(cols))
    for r in rows:
        print(",".join(str(r.get(c, "")) for c in cols))
    print(f"# written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
