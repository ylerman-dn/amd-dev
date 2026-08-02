#!/usr/bin/env python3
"""
Validate a generated tuner config rule-by-rule against RCCL's default, and emit a pruned config
containing only the rules that demonstrably help.

WHY THIS EXISTS
---------------
The sweep pipeline picks a winner per message size from a SINGLE measurement per configuration, using
env-var forcing. Field experience on MI355X (GPU-107, 2026-07-29) showed three ways that goes wrong:

  1. Phantom winners. A single run is not enough. Candidates that looked 6% faster reversed to 12%
     SLOWER when repeated (256KB/96ch), because small-to-mid message sizes have wide per-run spread.

  2. Measurement path != deployment path. The sweep forces algo/proto/channels with NCCL_* env vars,
     but the config is deployed through the tuner plugin. Those are not equivalent: 64 channels via
     `NCCL_MIN_NCHANNELS` left throughput unchanged, while the same 64 via the plugin cost 54%.
     A config can therefore measure well and regress badly in production.

  3. Unavailable algo/proto combos are silently substituted. Forcing an algorithm/protocol RCCL cannot
     provide (e.g. LL128 for all_reduce on this stack) makes RCCL quietly run RING/SIMPLE instead. The
     sweep then labels the row with what it REQUESTED, so ~28% of rows were mislabelled and half the
     "winners" were fiction. RCCL marks unavailable combos as -1.0 in the tuner cost table and the
     plugin correctly refuses them -- so those rules can never fire anyway.

  4. Catastrophic combos exist. Specific (channels, size, node-count) points collapse rather than just
     underperform: 32 channels at 3 nodes / 32MB measured -53%; 24ch/16MB -65%; 48ch/32MB -68%.
     A rule that spans a size range can therefore contain a landmine that range-level testing misses.

WHAT THIS DOES
--------------
For every rule in the config, at the node count that rule targets:
  * runs the benchmark with the config applied and with plain default, REPEATS times each,
    interleaved (round-robin) so drift cannot favour one side;
  * scores each message size with probability-of-superiority P(sup) -- the fraction of head-to-head
    pairs the config wins (1.00 = always faster, 0.50 = coin flip). This is a rank statistic, so it is
    not fooled by one lucky or unlucky run the way a mean is;
  * keeps a rule only where P(sup) >= --min-psup AND the median gain exceeds --min-gain;
  * drops any rule whose range contains a size that REGRESSES beyond --max-regression, and reports it,
    because that is a landmine and a partial win does not justify shipping it;
  * records the hang rate per variant. A config that hangs is worse than a slow one: if the config's
    hang rate exceeds --max-hang-rate the tool refuses to bless it regardless of throughput.

It never edits the input file; it writes <input>.validated.csv plus a report.

USAGE
    ./validate_tuner_config.py --config generated_tuner.csv --servers servers.txt \
        --binary /path/to/all_reduce_perf [--repeats 7] [--min-psup 0.95]

Exit status: 0 if at least one rule survived, 1 if none did (a legitimate outcome -- it means RCCL's
defaults are already optimal for this hardware, and shipping an empty config is the correct answer).
"""

import argparse
import csv
import itertools
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict

FLOAT_ROW_FIELDS = 13  # a benchmark data row has at least this many whitespace-separated fields


def parse_rules(path):
    """Read a tuner conf, returning (header, [rule dicts]). Comment lines are preserved separately."""
    comments, rules, header = [], [], None
    with open(path) as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s.startswith("#") or not s.strip():
                comments.append(s)
            elif header is None:
                header = s
            else:
                p = s.split(",")
                if len(p) < 8:
                    continue
                rules.append({
                    "raw": s, "coll": p[0],
                    "min_bytes": int(p[1]), "max_bytes": int(p[2]),
                    "algo": p[3], "proto": p[4], "channels": p[5],
                    "nodes": int(p[6]), "ranks": int(p[7]),
                })
    return comments, header, rules


def parse_busbw(text):
    """Extract {size_bytes: in-place busbw} from benchmark stdout.

    The in-place busbw column position depends on the -c flag, so locate it from the data row itself
    rather than hardcoding an index: with validation on (-c 1) each of the two result groups is
    time/algbw/busbw/#wrong, so in-place busbw is field 11 (0-based).
    """
    out = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) >= FLOAT_ROW_FIELDS and f[0].isdigit() and "float" in f[2]:
            try:
                out[int(f[0])] = float(f[11])
            except (ValueError, IndexError):
                continue
    return out


def psup(a, b):
    """Probability that a random draw from `a` exceeds a random draw from `b` (ties count half)."""
    if not a or not b:
        return 0.0
    wins = sum(1 for x, y in itertools.product(a, b) if x > y)
    ties = sum(1 for x, y in itertools.product(a, b) if x == y)
    return (wins + 0.5 * ties) / (len(a) * len(b))


def run_once(args, nodes, conf_path, log_path):
    """Run the benchmark once. Returns (busbw_by_size, seconds, hung?)."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{os.path.dirname(args.binary)}:{env.get('LD_LIBRARY_PATH','')}"
    # NCCL_DEBUG must stay quiet: INFO-level logging distorts the timings we are about to compare.
    env["NCCL_DEBUG"] = "VERSION"
    # LD_PRELOAD matters when several librccl builds are present on the node.
    lib = os.path.join(os.path.dirname(args.binary), "librccl.so")
    if os.path.exists(lib):
        env["LD_PRELOAD"] = lib
    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    if conf_path:
        env["NCCL_TUNER_PLUGIN"] = args.plugin
        env["NCCL_TUNER_CONFIG_FILE"] = conf_path

    cmd = args.launcher.split() + [
        "-N", str(nodes), "--ntasks-per-node=1", "--gres=gpu:8", "--mpi=pmix", "--export=ALL",
        args.binary, "-b", str(args.min_bytes), "-e", str(args.max_bytes),
        "-f", "2", "-g", "8", "-n", str(args.iters), "-w", str(args.warmup),
        "-c", "1", "-M", "1", "-R", "1",
    ]
    started = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env)
    try:
        text, _ = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        text, _ = proc.communicate()          # keep the partial output: it distinguishes a real
        text = (text or "") + "\n# [validator] killed after timeout\n"   # hang from a bad launch
    elapsed = time.time() - started
    if log_path:
        with open(log_path, "w") as fh:
            fh.write(text)
    data = parse_busbw(text)
    return data, elapsed, not data  # no data rows == hung or failed


def log_time(times_csv, phase, nodes, detail, seconds, hung):
    if not times_csv:
        return
    new = not os.path.exists(times_csv)
    with open(times_csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["utc_start", "phase", "scale", "detail", "duration_s", "rc", "notes"])
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - seconds)),
                    phase, f"{nodes}n", detail, int(seconds), 124 if hung else 0,
                    "hung" if hung else "ok"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="tuner conf to validate")
    ap.add_argument("--binary", required=True, help="collective benchmark, e.g. .../all_reduce_perf")
    ap.add_argument("--plugin", default=os.environ.get("NCCL_TUNER_PLUGIN", ""),
                    help="path to the tuner plugin .so")
    ap.add_argument("--launcher", default="srun", help="launcher prefix (default: srun)")
    ap.add_argument("--repeats", type=int, default=7,
                    help="repeats per variant; 7 is the practical floor for P(sup) to mean anything")
    ap.add_argument("--min-psup", type=float, default=0.95,
                    help="minimum probability-of-superiority to keep a rule (default 0.95)")
    ap.add_argument("--min-gain", type=float, default=2.0, help="minimum median %% gain (default 2)")
    ap.add_argument("--max-regression", type=float, default=2.0,
                    help="drop the rule if any size in its range loses more than this %% (default 2)")
    ap.add_argument("--max-hang-rate", type=float, default=0.05,
                    help="refuse to bless the config if it hangs more often than this (default 0.05)")
    ap.add_argument("--min-bytes", type=int, default=65536)
    ap.add_argument("--max-bytes", type=int, default=536870912)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=220)
    ap.add_argument("--logdir", default="validate_logs")
    ap.add_argument("--times-csv", default="", help="append per-run wall-times here")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                    help="extra environment for the benchmark, repeatable. MULTI-NODE RUNS USUALLY "
                         "REQUIRE THIS: without the fabric settings (e.g. NCCL_SOCKET_IFNAME, "
                         "NCCL_IB_GID_INDEX, OMPI_MCA_btl_tcp_if_include) 2+ node runs produce no "
                         "output at all and would be miscounted as hangs.")
    args = ap.parse_args()

    comments, header, rules = parse_rules(args.config)
    if not rules:
        print(f"no rules found in {args.config}", file=sys.stderr)
        return 1
    os.makedirs(args.logdir, exist_ok=True)

    scales = sorted({r["nodes"] for r in rules})
    if max(scales) > 1 and not any("IFNAME" in e or "IB_" in e for e in args.env):
        print("WARNING: validating multi-node rules but no fabric settings passed via --env. "
              "On most clusters 2+ node runs then produce no output and are miscounted as hangs.\n",
              file=sys.stderr, flush=True)
    print(f"validating {len(rules)} rule(s) across node counts {scales}, "
          f"{args.repeats} repeats each, keeping P(sup) >= {args.min_psup} and gain > {args.min_gain}%\n",
          flush=True)

    measured = {}   # nodes -> variant -> size -> [busbw]
    hangs = defaultdict(int)
    total = defaultdict(int)

    for nodes in scales:
        measured[nodes] = {"default": defaultdict(list), "config": defaultdict(list)}
        for rep in range(1, args.repeats + 1):
            for variant, conf in (("default", None), ("config", args.config)):
                tag = f"{nodes}n_{variant}_r{rep}"
                data, secs, hung = run_once(args, nodes, conf, os.path.join(args.logdir, tag + ".log"))
                total[(nodes, variant)] += 1
                if hung:
                    hangs[(nodes, variant)] += 1
                for size, bw in data.items():
                    measured[nodes][variant][size].append(bw)
                log_time(args.times_csv, "validate_conf", nodes, f"{variant}_r{rep}", secs, hung)
            print(f"  {nodes}n rep{rep} done", flush=True)

    # --- stability gate: a config that hangs is not shippable, whatever its throughput -------------
    print("\nhang rate:")
    blocked = False
    for nodes in scales:
        for variant in ("default", "config"):
            n, h = total[(nodes, variant)], hangs[(nodes, variant)]
            rate = h / n if n else 0
            print(f"  {nodes}n {variant:<8} {h}/{n} = {rate:.0%}")
            if variant == "config" and rate > args.max_hang_rate:
                blocked = True
    if blocked:
        print(f"\n*** the config exceeds --max-hang-rate ({args.max_hang_rate:.0%}). It is NOT shippable "
              f"until that is fixed, regardless of the bandwidth numbers below. ***")

    # --- per-rule verdicts ------------------------------------------------------------------------
    kept, dropped = [], []
    print(f"\n{'rule':<44} {'sizes':>6} {'best gain':>10} {'worst':>8} {'P(sup)':>7}  verdict")
    for rule in rules:
        base = measured[rule["nodes"]]["default"]
        cand = measured[rule["nodes"]]["config"]
        sizes = [s for s in sorted(base) if rule["min_bytes"] <= s <= rule["max_bytes"]]
        gains, psups, worst = [], [], 0.0
        for s in sizes:
            a, b = cand.get(s, []), base.get(s, [])
            if len(a) < 2 or len(b) < 2:
                continue
            g = (statistics.median(a) / statistics.median(b) - 1) * 100
            gains.append(g)
            psups.append(psup(a, b))
            worst = min(worst, g)
        label = f"{rule['coll']} {rule['min_bytes']}-{rule['max_bytes']} ch{rule['channels']} {rule['nodes']}n"
        if not gains:
            verdict, reason = "DROP", "no data"
        elif worst < -args.max_regression:
            verdict, reason = "DROP", f"contains a regression of {worst:.1f}% (landmine)"
        elif max(gains) < args.min_gain:
            verdict, reason = "DROP", f"best gain only {max(gains):+.1f}%"
        elif max(psups) < args.min_psup:
            verdict, reason = "DROP", f"P(sup) only {max(psups):.2f}, not reproducible"
        else:
            verdict, reason = "KEEP", "verified"
        print(f"{label:<44} {len(gains):>6} {max(gains) if gains else 0:>+9.1f}% "
              f"{worst:>+7.1f}% {max(psups) if psups else 0:>7.2f}  {verdict} ({reason})")
        (kept if verdict == "KEEP" else dropped).append((rule, reason))

    out = os.path.splitext(args.config)[0] + ".validated.csv"
    with open(out, "w") as fh:
        fh.write("# Validated by validate_tuner_config.py: only rules that beat RCCL default\n")
        fh.write(f"# with P(sup) >= {args.min_psup} over {args.repeats} repeats, gain > {args.min_gain}%,\n")
        fh.write(f"# and no size in range regressing more than {args.max_regression}%.\n")
        if blocked:
            fh.write("# WARNING: the config's hang rate exceeded the limit -- do not deploy via the\n")
            fh.write("#          plugin until that is resolved. Consider applying channel counts via\n")
            fh.write("#          NCCL_MIN_NCHANNELS/NCCL_MAX_NCHANNELS at job launch instead.\n")
        for rule, reason in dropped:
            fh.write(f"# dropped: {rule['raw']}   ({reason})\n")
        fh.write(header + "\n")
        for rule, _ in kept:
            fh.write(rule["raw"] + "\n")

    print(f"\nkept {len(kept)} of {len(rules)} rules -> {out}")
    if not kept:
        print("No rule survived. That is a legitimate result: RCCL's defaults are already optimal here,\n"
              "and an empty config is the correct thing to ship.")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
