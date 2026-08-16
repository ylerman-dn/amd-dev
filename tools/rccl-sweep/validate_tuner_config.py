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


def split_passing_runs(per_size, min_gain, min_psup, max_regression):
    """Split a mixed rule range into the contiguous runs of sizes that individually pass.

    `per_size` is [(size_bytes, gain_pct, psup)] ascending by size, covering only the sizes that
    actually have data. A size passes on the same thresholds the whole-rule verdict uses.

    Returns (runs, excluded_sizes) where runs is a list of lists of the passing tuples. Returns
    ([], []) when splitting would not help -- nothing passes, or everything does (in which case the
    caller's normal KEEP path already handles it).

    Why contiguous: a tuner rule is a byte range, so only an unbroken span of sizes can become one
    rule. A failing size in the middle genuinely has to break the range in two.
    """
    ok = {s for s, g, p in per_size
          if g >= min_gain and p >= min_psup and g >= -max_regression}
    if not ok or len(ok) == len(per_size):
        return [], []
    runs, cur = [], []
    for s, g, p in per_size:
        if s in ok:
            cur.append((s, g, p))
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs, [s for s, _, _ in per_size if s not in ok]


def self_test():
    """Unit-test split_passing_runs against the cases that motivated it. Returns an exit code."""
    K, M = 1024, 1024 * 1024
    cases = [
        # (name, per_size, expected number of runs, expected excluded sizes)
        ("3n 256K-1M ch16 (real landmine: +8.9% then -5.0%)",
         [(256*K, +8.9, 1.00), (512*K, -5.0, 0.10), (1*M, -3.8, 0.14)], 1, [512*K, 1*M]),
        ("3n 256M-512M ch32 (real landmine: +12.1% then -2.5%)",
         [(256*M, +12.1, 1.00), (512*M, -2.5, 0.20)], 1, [512*M]),
        ("failure in the MIDDLE must break the range in two",
         [(64*K, +5.0, 1.0), (128*K, -4.0, 0.1), (256*K, +6.0, 1.0)], 2, [128*K]),
        ("all sizes pass -> splitter declines, caller's KEEP path applies",
         [(64*K, +5.0, 1.0), (128*K, +6.0, 1.0)], 0, []),
        ("no size passes -> stays dropped",
         [(64*K, -5.0, 0.1), (128*K, -4.0, 0.1)], 0, []),
        ("good gain but P(sup) too low must NOT be smuggled in",
         [(64*K, +9.0, 0.60), (128*K, +5.0, 1.00)], 1, [64*K]),
        ("gain below --min-gain is excluded even when reproducible",
         [(64*K, +0.5, 1.00), (128*K, +5.0, 1.00)], 1, [64*K]),
    ]
    bad = 0
    for name, per_size, exp_runs, exp_excl in cases:
        runs, excl = split_passing_runs(per_size, 2.0, 0.95, 2.0)
        ok = len(runs) == exp_runs and excl == exp_excl
        bad += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print(f"         expected {exp_runs} run(s), excluded {exp_excl}")
            print(f"         got      {len(runs)} run(s), excluded {excl}")
    print(f"\n{len(cases) - bad}/{len(cases)} passed")
    return 1 if bad else 0


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


def preflight(args, scales):
    """Measure whether the machine can currently produce a decidable answer, before spending an A/B.

    Runs the DEFAULT configuration a handful of times at each scale and looks at how much its own
    repeats disagree. Nothing is being compared here -- identical runs should give identical numbers,
    so any large spread is the environment, not the config.

    Why this exists: on 2026-08-16 a 2-node A/B ran to completion and reported "0 of 11 rules kept,
    RCCL's defaults are already optimal" when the real cause was a co-tenant job saturating the
    shared fabric -- the same configuration measured 1.48-25.36 GB/s at 1M. A five-run probe of the
    same nodes showed 210% spread at 8M in 40 seconds. That is the cost of knowing, versus ~5 minutes
    per scale to produce a result that has to be thrown away.

    Returns True when every scale looks decidable.
    """
    print(f"preflight: {args.preflight} default runs per scale, checking the machine can decide "
          f"anything before spending an A/B\n", flush=True)
    ok = True
    for nodes in scales:
        seen = defaultdict(list)
        for i in range(1, args.preflight + 1):
            data, _, hung = run_once(args, nodes, None,
                                     os.path.join(args.logdir, f"{nodes}n_preflight_r{i}.log"))
            if hung:
                print(f"  {nodes}n preflight run {i} HUNG", flush=True)
                ok = False
                continue
            for size, bw in data.items():
                seen[size].append(bw)
        worst, worst_size = 0.0, None
        for size, vals in seen.items():
            if len(vals) >= 2 and min(vals) > 0:
                sp = (max(vals) - min(vals)) / min(vals) * 100
                if sp > worst:
                    worst, worst_size = sp, size
        if not seen:
            print(f"  {nodes}n: no data from any preflight run")
            ok = False
        elif worst > args.max_arm_spread:
            print(f"  {nodes}n: WORST SPREAD {worst:.0f}% at {worst_size} bytes "
                  f"(limit {args.max_arm_spread:.0f}%) -- NOT usable")
            ok = False
        else:
            print(f"  {nodes}n: worst spread {worst:.1f}% -- usable")
    if not ok:
        print("\n*** PREFLIGHT FAILED. Repeats of the SAME configuration disagree by more than the\n"
              "    effect an A/B would be measuring, so any verdict would be noise.\n"
              "    Usual causes: a co-tenant saturating the shared fabric (check squeue -p XAI),\n"
              "    GPUs on idle clocks (raise --warmup-runs), or another run competing for these\n"
              "    nodes. Fix the cause and retry; pass --no-preflight to override. ***")
    else:
        print("preflight OK\n")
    return ok


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
    # GPU-107: INFO to a FILE is free -- measured mean -0.06% across 18 sizes with
    # min/max ranges overlapping 18/18 (results-tuning/2026-08-03-5-lognoise/). Only
    # INFO on *stdout* costs (up to 4% at small sizes). We need INFO because it is the
    # only proof that the tuner plugin actually fired ("TUNER/Plugin: Applied config")
    # and the only source of real channel counts. Both arms get it identically, so any
    # residual bias cannot create a fake winner.
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_SUBSYS"] = "INIT,TUNING,ENV"
    if log_path:
        env["NCCL_DEBUG_FILE"] = os.path.splitext(log_path)[0] + "_dbg_%p.log"
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
        "-N", str(nodes),
    ]
    # GPU-107: run inside the existing allocation rather than requesting a new one.
    if args.jobid:
        cmd += [f"--jobid={args.jobid}"]
    if args.nodelist:
        cmd += [f"--nodelist={args.nodelist}"]
    cmd += [
        # GPU-107: 8 ranks x 1 GPU, NOT 1 rank x 8 GPUs. Every measurement in
        # results-tuning/ uses 8x1; the original 1x8 here would have produced an A/B
        # that is not comparable to the sweep that generated the config.
        f"--ntasks-per-node={args.ranks_per_node}", "--gres=gpu:8", "--mpi=pmix", "--export=ALL",
        args.binary, "-b", str(args.min_bytes), "-e", str(args.max_bytes),
        "-f", "2", "-g", str(args.gpus_per_rank),
        "-n", str(args.iters), "-w", str(args.warmup),
        # GPU-107: -A is --output_algo_proto_channels on this build. -M here is
        # --memory_report and -R is --local_register (which silently changes
        # performance), so the original "-M 1 -R 1" both failed to report selection
        # and perturbed the measurement.
        "-c", "1", "-A", "1",
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

    # GPU-107: for a config arm, prove the plugin loaded AND applied a rule. A config
    # that silently fails to load would otherwise be A/B'd against itself and reported
    # as "no difference".
    if conf_path and args.plugin_must_fire:
        import glob as _glob
        applied = False
        for f in _glob.glob(os.path.splitext(log_path)[0] + "_dbg_*.log"):
            try:
                if "TUNER/Plugin: Applied config" in open(f, errors="ignore").read():
                    applied = True
                    break
            except OSError:
                pass
        if not applied:
            print(f"    !! plugin did NOT apply any rule for {os.path.basename(log_path)} "
                  f"-- treating as failed rather than as 'no difference'")
            return {}, elapsed, True

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
    # GPU-107 additions
    ap.add_argument("--jobid", default="", help="run inside this existing Slurm allocation")
    ap.add_argument("--nodelist", default="", help="pin to these nodes (comma separated)")
    ap.add_argument("--ranks-per-node", type=int, default=8,
                    help="ranks per node; 8 matches every measurement in results-tuning/")
    ap.add_argument("--gpus-per-rank", type=int, default=1,
                    help="-g value; 1 with 8 ranks/node = 8 GPUs, matching the sweep")
    ap.add_argument("--plugin-must-fire", action="store_true",
                    help="fail unless the INFO log proves the tuner plugin applied the config")
    ap.add_argument("--repeats", type=int, default=7,
                    help="repeats per variant; 7 is the practical floor for P(sup) to mean anything")
    ap.add_argument("--max-arm-spread", type=float, default=25.0,
                    help="a (size, arm) point is 'noisy' when its own repeats span more than this "
                         "%% of their minimum (default 25). Clean 1-node runs sit near 1%%.")
    ap.add_argument("--max-noisy-fraction", type=float, default=0.20,
                    help="if more than this fraction of points are noisy, the run is reported as "
                         "NOT VALID and exits 2 instead of issuing verdicts (default 0.20)")
    ap.add_argument("--preflight", type=int, default=3,
                    help="default-config runs per scale before the A/B, to check the machine can "
                         "decide anything at all (default 3). Repeats of one config that disagree "
                         "more than --max-arm-spread mean the environment is unusable. 0 disables.")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the preflight and run the A/B regardless")
    ap.add_argument("--warmup-runs", type=int, default=4,
                    help="discarded full runs before the measured repeats, to bring the GPUs off "
                         "idle clocks (default 4). Distinct from -w/--warmup, which is iterations "
                         "inside one launch and does not affect DVFS. 0 disables.")
    ap.add_argument("--min-psup", type=float, default=0.95,
                    help="minimum probability-of-superiority to keep a rule (default 0.95)")
    ap.add_argument("--min-gain", type=float, default=2.0, help="minimum median %% gain (default 2)")
    ap.add_argument("--max-regression", type=float, default=2.0,
                    help="drop the rule if any size in its range loses more than this %% (default 2)")
    ap.add_argument("--max-hang-rate", type=float, default=0.05,
                    help="refuse to bless the config if it hangs more often than this (default 0.05)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the range-splitting unit tests and exit (no cluster needed)")
    ap.add_argument("--split-ranges", action="store_true",
                    help="when a rule's range mixes passing and failing sizes, emit the passing "
                         "contiguous sub-ranges instead of dropping the whole rule. Off by default: "
                         "it changes which rules ship, so it must be asked for explicitly. Each "
                         "sub-range still has to clear --min-gain, --min-psup and --max-regression "
                         "on every size it covers.")
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
    # --self-test needs no cluster, no config and no binary, so short-circuit before argparse
    # enforces the required arguments.
    if "--self-test" in sys.argv:
        sys.exit(self_test())

    args = ap.parse_args()

    comments, header, rules = parse_rules(args.config)
    if not rules:
        print(f"no rules found in {args.config}", file=sys.stderr)
        return 1
    os.makedirs(args.logdir, exist_ok=True)

    scales = sorted({r["nodes"] for r in rules})
    # GPU-107: check the ACTUAL environment too, not just --env. srun runs with --export=ALL, so
    # fabric settings exported by the calling shell are inherited and are perfectly valid. Looking
    # only at --env made this warn on correctly-configured runs, which trains the reader to ignore it.
    _fabric_set = (any("IFNAME" in e or "IB_" in e for e in args.env)
                   or any(k.startswith("NCCL_IB_") or k == "NCCL_SOCKET_IFNAME" for k in os.environ))
    if max(scales) > 1 and not _fabric_set:
        print("WARNING: validating multi-node rules but no fabric settings passed via --env. "
              "On most clusters 2+ node runs then produce no output and are miscounted as hangs.\n",
              file=sys.stderr, flush=True)
    print(f"validating {len(rules)} rule(s) across node counts {scales}, "
          f"{args.repeats} repeats each, keeping P(sup) >= {args.min_psup} and gain > {args.min_gain}%\n",
          flush=True)

    if args.preflight and not args.no_preflight:
        if not preflight(args, scales):
            return 3

    measured = {}   # nodes -> variant -> size -> [busbw]
    hangs = defaultdict(int)
    total = defaultdict(int)

    for nodes in scales:
        measured[nodes] = {"default": defaultdict(list), "config": defaultdict(list)}

        # --- clock warm-up ------------------------------------------------------------------------
        # The GPUs idle at ~195 MHz sclk (peak ~2400) and each repeat is a separate srun, so the
        # first few measured repeats run on a GPU that is still ramping. -w/--warmup does NOT cover
        # this: those are iterations inside one launch, microseconds at small sizes, which warm the
        # caches, not the clocks. DVFS needs seconds of sustained load.
        #
        # Measured 2026-08-16, 1 node, fresh allocation (results-tuning/2026-08-16-2-ab1node/):
        #   4M default  r1=92.8 r2=91.1 r3=88.4 r4=90.1 | r5=128.1 r6=128.1 r7=127.8
        # The step appears in BOTH arms at every size, so it does not bias which arm wins -- but it
        # inflates each arm's own spread to ~40%, the arms overlap, and P(sup) (a rank statistic)
        # collapses below the keep threshold. It rejected every rule in that run.
        #
        # 2026-08-04 did not hit this only because its A/B followed a 20-minute sweep on the same
        # node: all 7 repeats were flat to 0.6%, and today's r5-r7 reproduce those values exactly.
        #
        # These runs are discarded, never recorded, and cost ~1 min per scale.
        for w in range(1, args.warmup_runs + 1):
            tag = f"{nodes}n_warmup_r{w}"
            _, secs, hung = run_once(args, nodes, None, os.path.join(args.logdir, tag + ".log"))
            log_time(args.times_csv, "warmup", nodes, f"warmup_r{w}", secs, hung)
            print(f"  {nodes}n warm-up {w}/{args.warmup_runs} done (discarded)", flush=True)

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
        per_size = []          # (size, gain, psup) for the sizes that actually have data
        for s in sizes:
            a, b = cand.get(s, []), base.get(s, [])
            if len(a) < 2 or len(b) < 2:
                continue
            g = (statistics.median(a) / statistics.median(b) - 1) * 100
            p = psup(a, b)
            gains.append(g)
            psups.append(p)
            per_size.append((s, g, p))
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

        # --- range splitting ----------------------------------------------------------------------
        # generate_tuner_config.py merges adjacent sizes that share settings, so one rule can span a
        # real win and a real regression. Dropping the whole rule is correct but throws the win away:
        # at 3 nodes it discarded +8.9% @256K and +12.1% @256M. With --split-ranges, a mixed rule is
        # instead narrowed to the contiguous runs of sizes that individually pass, and the failing
        # sizes fall through to RCCL's default. Each emitted sub-rule is still backed by per-size
        # measurements at the same thresholds -- nothing is kept that was not measured to pass.
        if verdict == "DROP" and args.split_ranges and per_size:
            runs, excluded = split_passing_runs(
                per_size, args.min_gain, args.min_psup, args.max_regression)
            if runs:
                for run in runs:
                    lo, hi = run[0][0], run[-1][0]
                    # widen to the original bounds where nothing failed outside the run
                    p_raw = rule["raw"].split(",")
                    p_raw[1], p_raw[2] = str(lo), str(hi)
                    sub = dict(rule, raw=",".join(p_raw), min_bytes=lo, max_bytes=hi)
                    sub_label = f"  ↳ split {lo}-{hi} ch{rule['channels']} {rule['nodes']}n"
                    bg = max(g for _, g, _ in run)
                    bp = max(p for _, _, p in run)
                    wg = min(g for _, g, _ in run)
                    print(f"{sub_label:<44} {len(run):>6} {bg:>+9.1f}% {wg:>+7.1f}% {bp:>7.2f}  "
                          f"KEEP (split from a mixed range)")
                    kept.append((sub, f"split from {rule['min_bytes']}-{rule['max_bytes']}: {reason}"))
                dropped.append((rule, f"{reason} -- SPLIT: kept {len(runs)} sub-range(s), "
                                      f"excluded sizes {excluded}"))
                continue

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

    # --- validity gate: could this run decide anything at all? ------------------------------------
    # "0 rules kept" has two very different causes and the old wording asserted the wrong one:
    #   (a) the default genuinely wins        -> an empty config is the right answer
    #   (b) the measurement could not decide  -> the run is void and must be repeated
    # Both were printed as "RCCL's defaults are already optimal here". On 2026-08-16 that claim was
    # made for a 2-node run whose config arm measured 1.48-25.36 GB/s at 1M (co-tenant saturating
    # the shared fabric), and for a 1-node run on cold GPUs that kept 0 of 11 -- the same config
    # kept 7 of 11 once warmed. Neither was evidence about RCCL's defaults.
    #
    # A rule cannot be judged when an arm's own repeats scatter more than the gap being tested for,
    # so report spread and refuse to editorialise above the threshold.
    spreads = []
    for nodes in scales:
        for variant in ("default", "config"):
            for size, vals in measured[nodes][variant].items():
                if len(vals) >= 2 and min(vals) > 0:
                    spreads.append(((max(vals) - min(vals)) / min(vals) * 100, nodes, variant, size))
    worst_spread = max(spreads)[0] if spreads else 0.0
    noisy = [s for s in spreads if s[0] > args.max_arm_spread]

    print(f"\nwithin-arm spread: worst {worst_spread:.1f}%, "
          f"{len(noisy)} of {len(spreads)} (size, arm) points over {args.max_arm_spread:.0f}%")
    invalid = len(noisy) > len(spreads) * args.max_noisy_fraction

    print(f"\nkept {len(kept)} of {len(rules)} rules -> {out}")

    if invalid:
        wp, wn, wv, ws = max(spreads)
        print(f"\n*** RUN NOT VALID. {len(noisy)} of {len(spreads)} points exceed "
              f"{args.max_arm_spread:.0f}% within-arm spread (worst {wp:.0f}% at "
              f"{ws} bytes, {wn}n {wv}).\n"
              f"    Repeats of the SAME configuration disagree by more than the effect being\n"
              f"    measured, so neither KEEP nor DROP above means anything. Do not read this as\n"
              f"    a statement about RCCL's defaults.\n"
              f"    Usual causes: a co-tenant saturating the shared fabric (check squeue -p XAI),\n"
              f"    GPUs still on idle clocks (raise --warmup-runs), or two runs competing for the\n"
              f"    same nodes. Fix the cause and repeat. ***")
        return 2

    if not kept:
        print("No rule survived, and the run is measurement-valid: RCCL's defaults are already\n"
              "optimal here, and an empty config is the correct thing to ship.")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
