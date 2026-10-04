#!/usr/bin/env python3
"""Rule hit-map for one in-model server (GPU-107, 2026-09-09).

Reads the server's RCCL logs (NCCL_DEBUG=INFO with the TUNING subsystem on, one file per rank)
and reports, per conf rule, how many times the tuner plugin applied it, plus every all_reduce
size RCCL executed that matched NO rule. Counts are summed over the 8 rank logs (each rank logs
its own calls, so divide by ranks for per-GPU counts).

With CUDA graphs on, decode collectives are captured once per batch size and replayed, so the
counts there are capture-time counts, not per-step call counts; prefill is not captured and
counts per call. With graphs off every call counts.

Usage:
  infer_hitmap.py --logs <server>/logs --conf <conf.csv> [-o hitmap.csv]
Output CSV columns: kind (rule|miss), min_bytes, max_bytes, algorithm, protocol, channels, hits, executed_channels
"""
import argparse
import csv
import glob
import gzip
import os
import re
import sys
from collections import Counter

APPLIED = re.compile(r"Applied config for collType=(\w+), bytes=(\d+),.*algo=(\w+), proto=(\w+), channels=(\w+)")
CALL = re.compile(r"(\w+): (\d+) Bytes -> Algo (\w+) proto (\w+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="dir with rccl.*.log (one per rank)")
    ap.add_argument("--conf", required=True, help="tuner conf the server ran with")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    rules = []
    for r in csv.DictReader(l for l in open(args.conf) if not l.startswith("#")):
        rules.append((int(r["min_bytes"]), int(r["max_bytes"]), r["algorithm"], r["protocol"], r["channels"]))
    applied = Counter()           # (bytes, algo, proto, ch) -> hits
    calls = Counter()             # (coll, bytes) -> calls
    exec_ch = {}                  # (coll, bytes) -> Counter of executed channel counts
    files = glob.glob(os.path.join(args.logs, "*.log")) + glob.glob(os.path.join(args.logs, "*.log.gz"))
    for f in files:
        opener = gzip.open if f.endswith(".gz") else open
        with opener(f, "rt", errors="ignore") as fh:
            for line in fh:
                m = APPLIED.search(line)
                if m:
                    applied[(int(m.group(2)), m.group(3), m.group(4), m.group(5))] += 1
                    continue
                m = CALL.search(line)
                if m:
                    k = (m.group(1), int(m.group(2)))
                    calls[k] += 1
                    exec_ch.setdefault(k, Counter())[int(m.group(6)) - int(m.group(5)) + 1] += 1
    rows = []
    covered = set()
    def match(rule, key):
        lo, hi, algo, proto, ch = rule
        b, a, p, c = key
        # a hit is the plugin applying THIS rule: size in range and the rule's own algo/proto/channels
        # (so servers that hosted several arms, e.g. paired detect runs, attribute hits per arm)
        return lo <= b <= hi and a.lower() == algo.lower() and p.lower() == proto.lower() and \
            (str(ch) == str(c) or (str(ch) == "-1" and c == "default"))

    for rule in rules:
        lo, hi, algo, proto, ch = rule
        hits = sum(n for key, n in applied.items() if match(rule, key))
        sizes = sorted(key[0] for key in applied if match(rule, key))
        for b in sizes:
            covered.add(b)
        ex = Counter()
        for (coll, b), cnt in exec_ch.items():
            if coll.lower() == "allreduce" and lo <= b <= hi:
                ex.update(cnt)
        rows.append(dict(kind="rule", min_bytes=lo, max_bytes=hi, algorithm=algo, protocol=proto, channels=ch,
                         hits=hits, sizes_hit=";".join(str(s) for s in sorted(set(sizes))),
                         executed_channels=";".join(f"{c}x{n}" for c, n in ex.most_common(3))))
    for (coll, b), cnt in sorted(calls.items(), key=lambda kv: -kv[1]):
        if coll.lower() != "allreduce":
            continue
        if any(match(rule, (b, rule[2], rule[3], rule[4])) for rule in rules):
            continue  # size covered by a rule of this conf
        rows.append(dict(kind="miss", min_bytes=b, max_bytes=b, algorithm="", protocol="", channels="",
                         hits=cnt, sizes_hit=str(b),
                         executed_channels=";".join(f"{c}x{n}" for c, n in exec_ch[(coll, b)].most_common(3))))
    cols = ["kind", "min_bytes", "max_bytes", "algorithm", "protocol", "channels", "hits", "sizes_hit", "executed_channels"]
    out = args.output or os.path.join(os.path.dirname(args.logs.rstrip("/")), "hitmap.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"# {len(files)} rank logs, {sum(applied.values())} applied lines, {sum(calls.values())} collective call lines")
    print(",".join(cols))
    for r in rows:
        print(",".join(str(r[c]) for c in cols))
    print(f"# written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
