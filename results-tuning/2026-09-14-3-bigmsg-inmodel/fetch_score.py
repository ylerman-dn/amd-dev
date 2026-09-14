#!/usr/bin/env python3
"""Campaign B' fetch + score.

fetch:  python3 fetch_score.py fetch <tag> <node> <jobid>
        pulls /data/ylerman/bigmsg-inmodel-2026-09-14/<tag> (bench logs, hit-maps, pick.txt, receipts; no rccl logs, no server.log) into ./<tag>/
score:  python3 fetch_score.py score [<tag> ...]
        per <tag>/b<batch>: search medians per arm (input tok/s, mean TTFT ms, output tok/s) from search/s<tag>_<arm>/bench_round*.log,
        A/B medians + P(sup) (default vs conf, stock vs default) from pass1/pass2/<arm>_*/bench_rep*.log (rep 1 dropped), or stockref.
        Writes <tag>/b<batch>/score.csv and prints a markdown table; the night page reads the csv files."""
import csv, glob, os, re, statistics, subprocess, sys
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/data/ylerman/bigmsg-inmodel-2026-09-14"
RX = {"in": re.compile(r"Input token throughput \(tok/s\):\s+([0-9.]+)"),
      "out": re.compile(r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
      "ttft": re.compile(r"Mean TTFT \(ms\):\s+([0-9.]+)")}


def parse(f):
    t = open(f, errors="ignore").read()
    r = {k: float(m.group(1)) for k, rx in RX.items() if (m := rx.search(t))}
    return r if len(r) == 3 else None


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def psup(a, b):
    """P(a > b) over all pairs."""
    pairs = [(x, y) for x, y in product(a, b)]
    return sum(x > y for x, y in pairs) / len(pairs) if pairs else float("nan")


def fetch(tag, node, job):
    cmd = (f"srun --overlap --jobid={job} -N1 -w {node} tar -C {ROOT} -cf - --exclude='*/logs' --exclude='server.log' {tag}")
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "amd-mi355x-1", cmd], stdin=subprocess.DEVNULL, capture_output=True)
    if p.returncode:
        sys.exit(f"fetch failed: {p.stderr.decode()[:300]}")
    subprocess.run(["tar", "-C", HERE, "-xf", "-"], input=p.stdout, check=True)
    print(f"fetched {tag} from {node}")


def score(tags):
    for tag in tags:
        for bdir in sorted(glob.glob(os.path.join(HERE, tag, "b*"))):
            rows = []
            # search
            for adir in sorted(glob.glob(os.path.join(bdir, "search", f"s{tag}_*"))):
                arm = os.path.basename(adir).split("_", 1)[1]
                if arm == "srv":
                    continue
                vals = [v for f in sorted(glob.glob(os.path.join(adir, "bench_round*.log"))) if (v := parse(f))]
                rows.append(dict(phase="search", arm=arm, n=len(vals), in_toks=med([v["in"] for v in vals]),
                                 ttft_ms=med([v["ttft"] for v in vals]), out_toks=med([v["out"] for v in vals]), psup_vs_default="", gain_pct_in=""))
            base = next((r for r in rows if r["arm"] == "none"), None)
            for r in rows:
                if base and base["in_toks"] and r["arm"] != "none":
                    r["gain_pct_in"] = f"{(r['in_toks'] / base['in_toks'] - 1) * 100:+.2f}"
            # A/B or stockref
            ab = {}
            for pas in ("pass1", "pass2", "stockref"):
                for adir in sorted(glob.glob(os.path.join(bdir, pas, "*_def")) + glob.glob(os.path.join(bdir, pas, "*_tun"))):
                    arm = os.path.basename(adir).split("_")[0]
                    vals = [v for f in sorted(glob.glob(os.path.join(adir, "bench_rep*.log")))[1:] if (v := parse(f))]  # rep 1 dropped
                    ab.setdefault((pas, arm), []).extend(vals)
            for (pas, arm), vals in sorted(ab.items()):
                d = ab.get((pas, "default"), [])
                ps = psup([v["in"] for v in vals], [v["in"] for v in d]) if d and arm != "default" else ""
                g = f"{(med([v['in'] for v in vals]) / med([v['in'] for v in d]) - 1) * 100:+.2f}" if d and arm != "default" else ""
                rows.append(dict(phase=pas, arm=arm, n=len(vals), in_toks=med([v["in"] for v in vals]), ttft_ms=med([v["ttft"] for v in vals]),
                                 out_toks=med([v["out"] for v in vals]), psup_vs_default=f"{ps:.3f}" if ps != "" else "", gain_pct_in=g))
            out = os.path.join(bdir, "score.csv")
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["phase", "arm", "n", "in_toks", "ttft_ms", "out_toks", "psup_vs_default", "gain_pct_in"])
                w.writeheader(); w.writerows(rows)
            print(f"\n## {tag} {os.path.basename(bdir)}  ({out})")
            print("| phase | arm | n | input tok/s | TTFT ms | output tok/s | gain vs default (in) | P(sup) |\n|---|---|---|---|---|---|---|---|")
            for r in rows:
                print(f"| {r['phase']} | {r['arm']} | {r['n']} | {r['in_toks']:.0f} | {r['ttft_ms']:.0f} | {r['out_toks']:.0f} | {r['gain_pct_in']} | {r['psup_vs_default']} |")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "fetch":
        fetch(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) >= 2 and sys.argv[1] == "score":
        score(sys.argv[2:] or [os.path.basename(d) for d in glob.glob(os.path.join(HERE, "*")) if os.path.isdir(d) and os.path.basename(d) not in ("confs",)])
    else:
        sys.exit(__doc__)
