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
import re
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



def load_sweep_env(path):
    """Read env_vars: from sweep_config.yaml -- the single source the sweep already uses.

    Before this, the sweep read that file automatically while the A/B took --env hand-typed on the
    command line, so the two ran under different environments. run_mn.sh (the reference the A/B was
    typed from) carries OMPI_MCA_* but none of the IONIC_* settings the yaml has, and that divergence
    was live from 2026-08-04 and invisible. On 2026-08-16 hand-typing also produced a wrong
    NCCL_IB_GID_INDEX and a truncated HCA list.

    Returns {} when the file or PyYAML is missing -- this must never be the reason an A/B cannot run.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
    except ImportError:
        print(f"  note: PyYAML not available, not loading env from {path}", file=sys.stderr)
        return {}
    try:
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
    except Exception as exc:
        print(f"  note: could not read {path}: {exc}", file=sys.stderr)
        return {}
    env = doc.get("env_vars") or {}
    # Never inherit these: the A/B sets its own logging, and PATH belongs to the caller.
    for drop in ("NCCL_DEBUG", "NCCL_DEBUG_FILE", "NCCL_DEBUG_SUBSYS", "PATH",
                 "NCCL_TOPO_DUMP_FILE", "NCCL_GRAPH_DUMP_FILE"):
        env.pop(drop, None)
    return {k: str(v) for k, v in env.items()}


def preflight(args, scales, rules=None):
    """Measure whether the machine can currently produce a decidable answer, before spending an A/B.

    With `rules` (--preflight-scope rules, agreed 2026-08-26): the decision is scoped to the sizes
    the config actually ships rules for, and a PARTIALLY noisy scale is used rather than refused --
    the chronically noisy sizes are returned as excluded, their rules get dropped with that reason,
    and every other size is judged normally. Chronic small-size scatter (broadcast 2n 4-16K: 22
    attempts refused; alltoall 1n 32K at 690% spread) otherwise vetoes sizes that measure cleanly.
    Refusal still happens when EVERY scoped size is noisy.

    Returns {scale: set(excluded sizes)} when usable (empty sets in unscoped mode), None when not.

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
          f"anything before spending an A/B"
          + (" (scoped to the config's rule sizes)" if rules else "") + "\n", flush=True)
    ok = True
    excluded = {n: set() for n in scales}

    def covered(size, nodes):
        if rules is None:
            return True
        return any(r["nodes"] == nodes and r["min_bytes"] <= size <= r["max_bytes"]
                   for r in rules)

    for nodes in scales:
        # GPU-107: warm THIS scale's nodes before judging them. Preflight used to run before any
        # warm-up, so on a freshly allocated node it measured the idle-clock ramp -- the very thing
        # --warmup-runs exists to remove -- and refused a machine that was fine. On 2026-08-16 the
        # 2-node scale passed only because those nodes had been warmed by earlier work, while the
        # 3-node scale added a cold node 6 and failed at 210% spread.
        for w in range(args.warmup_runs):
            run_once(args, nodes, None,
                     os.path.join(args.logdir, f"{nodes}n_preflight_warmup_r{w + 1}.log"))
        if args.warmup_runs:
            print(f"  {nodes}n warmed with {args.warmup_runs} discarded run(s)", flush=True)

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
        def worst_of(sample_sets):
            w, ws = 0.0, None
            for size, vals in sample_sets.items():
                if len(vals) >= 2 and min(vals) > 0:
                    sp = (max(vals) - min(vals)) / min(vals) * 100
                    if sp > w:
                        w, ws = sp, size
            return w, ws

        scoped = {sz: vals for sz, vals in seen.items() if covered(sz, nodes)}
        if rules is not None and seen and not scoped:
            # a scale with data but no rule coverage has nothing to judge
            print(f"  {nodes}n: no rule covers any measured size -- nothing to preflight")
            continue
        worst, worst_size = worst_of(scoped)
        if not seen:
            print(f"  {nodes}n: no data from any preflight run")
            ok = False
        elif worst > args.max_arm_spread:
            # single-outlier confirmation (user-approved 2026-08-25): most
            # refusals are ONE run photobombed by the ~30s telemetry scrape
            # while the other runs agree. If excluding exactly one run index
            # makes EVERY size pass, take one extra confirmation run; pass
            # only if the confirmed set is clean. Chronic scatter has no such
            # index and still fails; the rc=2 post-gate remains the backstop.
            n_runs = max(len(v) for v in scoped.values())
            outlier = None
            for i in range(n_runs):
                trial = {sz: [v for j, v in enumerate(vals) if j != i]
                         for sz, vals in scoped.items()}
                if worst_of(trial)[0] <= args.max_arm_spread:
                    outlier = i
                    break
            judged = scoped
            if outlier is not None:
                print(f"  {nodes}n: spread {worst:.0f}% at {worst_size} bytes "
                      f"caused by run {outlier + 1} alone -- taking one "
                      f"confirmation run")
                data, _, hung = run_once(
                    args, nodes, None,
                    os.path.join(args.logdir,
                                 f"{nodes}n_preflight_confirm.log"))
                if not hung:
                    confirmed = {sz: [v for j, v in enumerate(vals)
                                      if j != outlier]
                                 for sz, vals in scoped.items()}
                    for sz, bw in data.items():
                        if covered(sz, nodes):
                            confirmed.setdefault(sz, []).append(bw)
                    cw, cws = worst_of(confirmed)
                    if cw <= args.max_arm_spread:
                        print(f"  {nodes}n: confirmed clean (worst "
                              f"{cw:.1f}%) -- usable")
                        continue
                    print(f"  {nodes}n: confirmation run disagrees too "
                          f"(worst {cw:.0f}% at {cws} bytes)")
                    judged = confirmed
                else:
                    print(f"  {nodes}n: confirmation run HUNG")
            # scoped mode: exclude the chronically noisy sizes instead of
            # vetoing the sizes that measure cleanly; refuse only when
            # nothing in scope is measurable
            if rules is not None:
                noisy = {sz for sz, vals in judged.items()
                         if len(vals) >= 2 and min(vals) > 0 and
                         (max(vals) - min(vals)) / min(vals) * 100
                         > args.max_arm_spread}
                if len(noisy) < len(judged):
                    excluded[nodes] = noisy
                    print(f"  {nodes}n: {len(noisy)} of {len(judged)} scoped "
                          f"size(s) unmeasurable and EXCLUDED "
                          f"({sorted(noisy)}); the rest are judged normally "
                          f"-- usable")
                    continue
                print(f"  {nodes}n: every scoped size is noisy -- NOT usable")
                ok = False
            else:
                if outlier is None:
                    print(f"  {nodes}n: WORST SPREAD {worst:.0f}% at "
                          f"{worst_size} bytes (limit "
                          f"{args.max_arm_spread:.0f}%), no single outlier "
                          f"-- NOT usable")
                else:
                    print(f"  {nodes}n: NOT usable")
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
    return excluded if ok else None


RESOLUTION = 0.005  # benchmark prints busbw to 0.01: half a print unit


def mad_core(vals, z=3.5):
    """Repeats within z robust-z of the median (Iglewicz-Hoaglin). MAD=0
    falls back to the mean absolute deviation; scale floored at RESOLUTION."""
    med = statistics.median(vals)
    mad = statistics.median([abs(v - med) for v in vals])
    scale = 1.4826 * mad if mad > 0 else \
        1.2533 * statistics.mean([abs(v - med) for v in vals])
    scale = max(scale, RESOLUTION)
    return [v for v in vals if abs(v - med) / scale <= z]


def cv_pct(vals):
    m = statistics.mean(vals) if vals else 0
    return statistics.stdev(vals) / m * 100 if len(vals) > 1 and m else 0.0


def point_noisy_cv_core(vals, cv_limit=8.0, core_min=7):
    """The cv-core gate (agreed 2026-08-27): a point is noisy when its CV
    exceeds cv_limit AND its MAD core (>= core_min repeats) is not itself
    under the limit. Single transient dips are rescued via the core; chronic
    scatter and true bimodal collapse still flag."""
    if cv_pct(vals) <= cv_limit:
        return False
    core = mad_core(vals)
    return not (len(core) >= core_min and cv_pct(core) <= cv_limit)


def gate_values(vals, gate, core_min=7):
    """The repeats a rule's verdict is computed on: raw for the spread gate,
    the MAD core for cv-core (dips identified as outliers are discarded).
    A core smaller than core_min is not trusted - fall back to the raw
    repeats rather than judge on a hand-picked subset (2026-08-27: a 6-of-9
    core pushed a P(sup) 0.66 point to 0.99 - manufactured win)."""
    if gate != "cv-core" or len(vals) < 2:
        return vals
    core = mad_core(vals)
    return core if len(core) >= core_min else vals


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
    # GPU-107: sweep_config.yaml first, --env second -- so the A/B inherits exactly what the sweep
    # ran with, and an explicit --env still wins.
    env.update(getattr(args, "_sweep_env", {}) or {})
    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    if conf_path:
        if getattr(args, "arm_env", None):
            # env-arm A/B (alltoall): the tuner plugin is never consulted for
            # p2p-path collectives (enqueue.cc: alltoall becomes send/recv
            # tasks; getAlgoInfo/tuner runs only for coll tasks), so the ON
            # arm is the config's env vars instead of the plugin.
            for kv in args.arm_env:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    env[k] = v
        else:
            env["NCCL_TUNER_PLUGIN"] = args.plugin
            env["NCCL_TUNER_CONFIG_FILE"] = conf_path

    cname = None
    if getattr(args, "runtime", "container") == "container":
        # Container runtime (default since 2026-09-06): docker + mpirun-in-container on
        # the stock RCCL — the stack rules deploy on. NCCL env travels via mpirun -x
        # (comma-safe); host-path env assembled above is for the bare branch only.
        import container_run
        if nodes > 1:
            raise SystemExit("--runtime container supports 1 node only; use --runtime bare")
        cenv = {k: env[k] for k in ("NCCL_DEBUG", "NCCL_DEBUG_SUBSYS") if k in env}
        if log_path:
            # container-side path; host globs keep working because dirname(log_path)
            # is mounted at OUT_MOUNT
            base = os.path.splitext(os.path.basename(log_path))[0]
            cenv["NCCL_DEBUG_FILE"] = f"{container_run.OUT_MOUNT}/{base}_dbg_%p.log"
        # MSCCL executes small/mid sizes without consulting any tuner; default off so
        # the A/B measures the tunable path. Override with --env RCCL_MSCCL_ENABLE=1.
        cenv.setdefault("RCCL_MSCCL_ENABLE", "0")
        cenv.update(getattr(args, "_sweep_env", {}) or {})
        for kv in args.env:
            if "=" in kv:
                k, v = kv.split("=", 1)
                cenv[k] = v
        tuner_dir = conf_dir = None
        if conf_path:
            if getattr(args, "arm_env", None):
                for kv in args.arm_env:
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        cenv[k] = v
            else:
                tuner_dir = os.path.dirname(os.path.abspath(args.plugin))
                conf_dir = os.path.dirname(os.path.abspath(conf_path))
                cenv["NCCL_TUNER_PLUGIN"] = f"{container_run.TUNER_MOUNT}/{os.path.basename(args.plugin)}"
                cmount = container_run.TUNER_MOUNT if conf_dir == tuner_dir else container_run.CONF_MOUNT
                cenv["NCCL_TUNER_CONFIG_FILE"] = f"{cmount}/{os.path.basename(conf_path)}"
        test_argv = [f"{container_run.BIN_MOUNT}/{os.path.basename(args.binary)}",
                     "-b", str(args.min_bytes), "-e", str(args.max_bytes),
                     "-f", "2", "-g", str(args.gpus_per_rank),
                     "-n", str(args.iters), "-w", str(args.warmup),
                     "-c", "1", "-A", "1"]
        cname = f"rcclval-{os.getpid()}-{int(time.time()) % 100000}"
        cmd = container_run.build_docker_mpirun(
            name=cname, image=args.image,
            num_ranks=nodes * args.ranks_per_node,
            bin_dir=os.path.dirname(os.path.abspath(args.binary)),
            out_dir=os.path.dirname(os.path.abspath(log_path)) if log_path else ".",
            test_argv=test_argv, env_vars=cenv,
            tuner_dir=tuner_dir, conf_dir=conf_dir,
            drop_env=getattr(args, "drop_env", []))
        env = dict(os.environ)  # env for the docker CLIENT only
    else:
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
        if cname:
            # killing the docker CLIENT does not stop a GPU-hung container (2026-09-03)
            import container_run
            container_run.kill_container(cname)
        text, _ = proc.communicate()          # keep the partial output: it distinguishes a real
        text = (text or "") + "\n# [validator] killed after timeout\n"   # hang from a bad launch
    elapsed = time.time() - started
    if log_path:
        with open(log_path, "w") as fh:
            fh.write(text)
    data = parse_busbw(text)

    # GPU-107: for an env-arm config run, prove RCCL actually saw the vars: the ENV
    # debug subsystem echoes "<VAR> set by environment" per rank. Without this an
    # ON arm whose env got lost (srun --export comma truncation, a wrapper eating
    # vars) would be A/B'd against itself and reported as "no difference".
    if conf_path and getattr(args, "arm_env", None):
        import glob as _glob
        need = [kv.split("=", 1)[0] for kv in args.arm_env if "=" in kv]
        seen = set()
        for f in _glob.glob(os.path.splitext(log_path)[0] + "_dbg_*.log"):
            try:
                txt = open(f, errors="ignore").read()
            except OSError:
                continue
            for k in need:
                if f"{k} set by environment" in txt:
                    seen.add(k)
        missing = [k for k in need if k not in seen]
        if missing:
            print(f"    !! env arm vars NOT echoed by RCCL for {os.path.basename(log_path)}: "
                  f"{missing} -- treating as failed rather than as 'no difference'")
            return {}, elapsed, True

    # GPU-107: for a config arm, prove the plugin loaded AND applied a rule. A config
    # that silently fails to load would otherwise be A/B'd against itself and reported
    # as "no difference".
    if conf_path and args.plugin_must_fire and not getattr(args, "arm_env", None):
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
    ap.add_argument("--launcher", default="srun", help="launcher prefix (bare runtime only; default: srun)")
    # GPU-107 2026-09-06: container is the default runtime — rules deploy on the SGLang
    # container's stock RCCL + image env; bare metal measures a stack nothing runs on
    # (results-tuning/2026-09-03-2-stackcmp, -3-regime). bare remains for multi-node and
    # for validating team librccl fork builds.
    ap.add_argument("--runtime", choices=["container", "bare"], default="container",
                    help="where to run the benchmark (default: container; 1 node only)")
    ap.add_argument("--image", default="lmsysorg/sglang:v0.5.17-rocm720-mi35x",
                    help="docker image for --runtime container")
    ap.add_argument("--drop-env", action="append", default=[],
                    help="container runtime: REMOVE this image-provided env var from BOTH arms "
                         "(e.g. NCCL_MIN_NCHANNELS). Repeatable.")
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
    ap.add_argument("--sweep-config", default=None,
                    help="path to sweep_config.yaml whose env_vars: block supplies the base "
                         "environment (default: the copy beside this script). --env overrides it. "
                         "Use --no-sweep-env to ignore it entirely.")
    ap.add_argument("--no-sweep-env", action="store_true",
                    help="do not take any environment from sweep_config.yaml")
    ap.add_argument("--preflight", type=int, default=3,
                    help="default-config runs per scale before the A/B, to check the machine can "
                         "decide anything at all (default 3). Repeats of one config that disagree "
                         "more than --max-arm-spread mean the environment is unusable. 0 disables.")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the preflight and run the A/B regardless")
    ap.add_argument("--require-full-range", action="store_true",
                    help="drop a rule when ANY size in its range was preflight-excluded. For "
                         "unsplittable deployments (alltoall's env-arm is one global setting): an "
                         "unmeasurable size inside the range may hide a regression the setting "
                         "would still inflict, so the rule cannot be blessed.")
    ap.add_argument("--noise-gate", choices=["spread", "cv-core"], default="spread",
                    help="point-noise rule for the rc=2 void accounting and the verdict inputs. "
                         "'spread' (default, original): (max-min)/min > 25%%, verdicts on raw "
                         "repeats. 'cv-core' (agreed 2026-08-27): noisy when CV > 8%% AND the "
                         "MAD core (>=7 repeats within 3.5 robust-z) is not under 8%% CV; "
                         "verdicts are computed on the cores (transient dips discarded).")
    ap.add_argument("--replay-stats", default=None, metavar="PER_SIZE_STATS.CSV",
                    help="OFFLINE replay: take both arms' repeats (and executed combos) from a "
                         "stored per_size_stats.csv instead of running anything. No cluster "
                         "contact. Preflight is replaced by: a size is excluded when its "
                         "DEFAULT-arm point is noisy under the active gate (baseline "
                         "untrustworthy). Meant for re-judging voided runs under a new gate.")
    ap.add_argument("--preflight-scope", choices=["all", "rules"], default="all",
                    help="'rules' (agreed 2026-08-26): judge preflight spread only on sizes the "
                         "config ships rules for, and EXCLUDE chronically noisy sizes instead of "
                         "refusing the machine -- their rules drop with that reason, the clean "
                         "sizes get judged. Refusal still happens when every scoped size is noisy. "
                         "'all' is the original whole-ladder veto.")
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
    ap.add_argument("--arm-env", action="append", default=[], metavar="KEY=VAL",
                    help="env-arm A/B: the ON arm sets these variables instead of loading the "
                         "tuner plugin. For collectives the plugin cannot tune (alltoall runs on "
                         "the p2p path and never consults the tuner). The vars are verified in the "
                         "NCCL debug log ('<VAR> set by environment') on every ON run.")
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

    # Resolve the base environment once, before anything runs, and record it with the run.
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_config.yaml")
    cfg_path = args.sweep_config or default_cfg
    args._sweep_env = {} if args.no_sweep_env else load_sweep_env(cfg_path)
    if args._sweep_env:
        print(f"env: {len(args._sweep_env)} vars from {cfg_path}"
              + (f", {len(args.env)} overridden by --env" if args.env else ""))

    comments, header, rules = parse_rules(args.config)
    if not rules:
        print(f"no rules found in {args.config}", file=sys.stderr)
        return 1
    os.makedirs(args.logdir, exist_ok=True)

    scales = sorted({r["nodes"] for r in rules})
    # GPU-107: check the ACTUAL environment too, not just --env. srun runs with --export=ALL, so
    # fabric settings exported by the calling shell are inherited and are perfectly valid. Looking
    # only at --env made this warn on correctly-configured runs, which trains the reader to ignore it.
    # GPU-107: check for the specific keys the sweep declares, not merely "some IB_ variable".
    # The old check passed on 2026-08-16 while NCCL_IB_GID_INDEX was wrong, the HCA list truncated
    # and the whole OMPI_MCA_* block missing -- a guard that green-lights a broken run is worse than
    # none, because it teaches the reader to ignore it.
    _have = set(os.environ) | {e.split("=", 1)[0] for e in args.env if "=" in e} | set(args._sweep_env)
    _need = [k for k in (args._sweep_env or {})
             if k.startswith(("NCCL_IB_", "OMPI_MCA_", "IONIC_")) or k == "NCCL_SOCKET_IFNAME"]
    _missing = sorted(k for k in _need if k not in _have)
    if max(scales) > 1 and _missing:
        print(f"WARNING: {len(_missing)} fabric variable(s) the sweep uses are not set here: "
              f"{', '.join(_missing)}", file=sys.stderr)
    _fabric_set = any(k.startswith("NCCL_IB_") or k == "NCCL_SOCKET_IFNAME" for k in _have)
    if max(scales) > 1 and not _fabric_set:
        print("WARNING: validating multi-node rules but no fabric settings passed via --env. "
              "On most clusters 2+ node runs then produce no output and are miscounted as hangs.\n",
              file=sys.stderr, flush=True)
    print(f"validating {len(rules)} rule(s) across node counts {scales}, "
          f"{args.repeats} repeats each, keeping P(sup) >= {args.min_psup} and gain > {args.min_gain}%\n",
          flush=True)

    # Record the environment this run actually used, so the folder is self-describing.
    if args.logdir:
        os.makedirs(args.logdir, exist_ok=True)
        _probe_env = dict(args._sweep_env)
        for kv in args.env:
            if "=" in kv:
                k, v = kv.split("=", 1)
                _probe_env[k] = v
        with open(os.path.join(args.logdir, "resolved_env.txt"), "w") as fh:
            fh.write(f"# resolved by validate_tuner_config.py\n")
            fh.write(f"# base: {cfg_path if not args.no_sweep_env else '(none, --no-sweep-env)'}\n")
            fh.write(f"# overrides from --env: {len(args.env)}\n")
            for k in sorted(_probe_env):
                fh.write(f"{k}={_probe_env[k]}\n")

    def point_noisy(vals):
        if args.noise_gate == "cv-core":
            return point_noisy_cv_core(vals)
        return len(vals) >= 2 and min(vals) > 0 and \
            (max(vals) - min(vals)) / min(vals) * 100 > args.max_arm_spread

    pf_excluded = {n: set() for n in scales}
    replay_exec = {}
    if args.replay_stats:
        # offline: repeats and executed combos come from the stored sidecar
        measured = {n: {"default": defaultdict(list), "config": defaultdict(list)}
                    for n in scales}
        with open(args.replay_stats) as fh:
            for row in csv.DictReader(fh):
                n, s = int(row["nodes"]), int(row["size_bytes"])
                if n not in measured:
                    continue
                for variant, key in (("default", "def_runs"), ("config", "cfg_runs")):
                    vals = [float(x) for x in row[key].split(";") if x]
                    measured[n][variant][s] = vals
                    lbl = row.get("def_exec" if variant == "default" else "cfg_exec", "")
                    if lbl:
                        parts = lbl.split("/")
                        replay_exec.setdefault((n, variant), {})[s] = \
                            (parts[0], parts[1] if len(parts) > 1 else "",
                             parts[2] if len(parts) > 2 else "", None)
        for n in scales:
            pf_excluded[n] = {s for s, vals in measured[n]["default"].items()
                              if point_noisy(vals)}
            if pf_excluded[n]:
                print(f"replay: {n}n sizes excluded (default arm noisy under "
                      f"{args.noise_gate}): {sorted(pf_excluded[n])}")
        print(f"replay: repeats loaded from {args.replay_stats}, gate {args.noise_gate}\n")
    elif args.preflight and not args.no_preflight:
        pf = preflight(args, scales,
                       rules if args.preflight_scope == "rules" else None)
        if pf is None:
            return 3
        pf_excluded.update(pf)

    if not args.replay_stats:
        measured = {}   # nodes -> variant -> size -> [busbw]
    hangs = defaultdict(int)
    total = defaultdict(int)

    for nodes in (scales if not args.replay_stats else []):
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
        _already_warm = bool(args.preflight and not args.no_preflight and args.warmup_runs)
        for w in range(1, 0 if _already_warm else args.warmup_runs + 1):
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

    # --- per-size stats sidecar ---------------------------------------------------------------
    # everything the verdicts are computed from, per size: both arms' raw run
    # values, medians, min/max and P(sup). Presentation (spread, CI) is built
    # downstream from this file; the verdict logic below is unchanged.
    # executed truth per arm, from the runs' own NCCL_DEBUG logs
    # (merge_metrics.parse_exec_log; the -A 1 channel column is a plan, not
    # what ran - finding 01)
    import glob as _glob2
    from merge_metrics import parse_exec_log
    exec_truth = {}
    p2p_truth = {}
    for nodes in scales:
        for variant in ("default", "config"):
            paths = _glob2.glob(os.path.join(
                args.logdir, f"{nodes}n_{variant}_r*_dbg_*.log"))
            exec_truth[(nodes, variant)] = parse_exec_log(paths)
            if not exec_truth[(nodes, variant)]:
                # p2p-path collectives (alltoall) print no per-size
                # "Bytes -> Algo" lines; the executed channel truth is the
                # comm's p2p channel count, one value for all sizes.
                counts = []
                for f in paths:
                    try:
                        counts += [int(m) for m in re.findall(
                            r"p2p channels:(\d+)",
                            open(f, errors="ignore").read())]
                    except OSError:
                        pass
                if counts:
                    n_p2p = max(set(counts), key=counts.count)
                    p2p_truth[(nodes, variant)] = ("P2P", "-", n_p2p, None)

    stats_path = os.path.join(args.logdir, "per_size_stats.csv")
    with open(stats_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nodes", "size_bytes", "def_median", "cfg_median",
                    "gain_pct", "psup", "def_min", "def_max", "cfg_min",
                    "cfg_max", "n_def", "n_cfg", "def_runs", "cfg_runs",
                    "def_exec", "cfg_exec", "cfg_applied_ch", "ch_trimmed"])
        for nodes in scales:
            base = measured[nodes]["default"]
            cand = measured[nodes]["config"]
            for s2 in sorted(set(base) | set(cand)):
                a = gate_values(cand.get(s2, []), args.noise_gate)
                b = gate_values(base.get(s2, []), args.noise_gate)
                if not a or not b:
                    continue
                dm = statistics.median(b)
                de = exec_truth.get((nodes, "default"), {}).get(s2) \
                    or replay_exec.get((nodes, "default"), {}).get(s2) \
                    or p2p_truth.get((nodes, "default"))
                ce = exec_truth.get((nodes, "config"), {}).get(s2) \
                    or replay_exec.get((nodes, "config"), {}).get(s2) \
                    or p2p_truth.get((nodes, "config"))
                d_lbl = f"{de[0]}/{de[1]}/{de[2]}" if de else ""
                c_lbl = f"{ce[0]}/{ce[1]}/{ce[2]}" if ce else ""
                app = ce[3] if ce else None
                trimmed = ("yes" if ce and app is not None and app != ce[2]
                           else "")
                w.writerow([nodes, s2, round(dm, 4),
                            round(statistics.median(a), 4),
                            round((statistics.median(a) / dm - 1) * 100, 3)
                            if dm else "",
                            round(psup(a, b), 4),
                            min(b), max(b), min(a), max(a), len(b), len(a),
                            ";".join(str(v) for v in b),
                            ";".join(str(v) for v in a),
                            d_lbl, c_lbl,
                            "" if app is None else app, trimmed])
    print(f"\nper-size stats: {stats_path}")

    # --- per-rule verdicts ------------------------------------------------------------------------
    kept, dropped = [], []
    print(f"\n{'rule':<44} {'sizes':>6} {'best gain':>10} {'worst':>8} {'P(sup)':>7}  verdict")
    for rule in rules:
        base = measured[rule["nodes"]]["default"]
        cand = measured[rule["nodes"]]["config"]
        _excl = pf_excluded.get(rule["nodes"], set())
        in_range = [s for s in sorted(base) if rule["min_bytes"] <= s <= rule["max_bytes"]]
        sizes = [s for s in in_range if s not in _excl]
        if args.require_full_range and len(sizes) < len(in_range):
            label = f"{rule['coll']} {rule['min_bytes']}-{rule['max_bytes']} ch{rule['channels']} {rule['nodes']}n"
            _hidden = sorted(set(in_range) - set(sizes))
            print(f"{label:<44} {0:>6} {0:>+9.1f}% {0:>+7.1f}% {0:>7.2f}  DROP "
                  f"(unsplittable rule, unmeasurable sizes in range: {_hidden})")
            dropped.append((rule, f"unsplittable rule, unmeasurable sizes in range: {_hidden}"))
            continue
        if in_range and not sizes:
            label = f"{rule['coll']} {rule['min_bytes']}-{rule['max_bytes']} ch{rule['channels']} {rule['nodes']}n"
            print(f"{label:<44} {0:>6} {0:>+9.1f}% {0:>+7.1f}% {0:>7.2f}  DROP "
                  f"(preflight: every size in range unmeasurable)")
            dropped.append((rule, "preflight: every size in range unmeasurable"))
            continue
        gains, psups, worst = [], [], 0.0
        per_size = []          # (size, gain, psup) for the sizes that actually have data
        for s in sizes:
            a = gate_values(cand.get(s, []), args.noise_gate)
            b = gate_values(base.get(s, []), args.noise_gate)
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
    # validity pre-check: never write verdicts for a run the gate below voids
    # sizes the scoped preflight excluded are already vetoed per rule; letting
    # their (known chronic) scatter also trip the whole-run rc=2 void would
    # re-veto the clean sizes the scoping exists to save
    _spreads_pre = []
    for _n in scales:
        for _v in ("default", "config"):
            for _sz, _vals in measured[_n][_v].items():
                if _sz in pf_excluded.get(_n, set()):
                    continue
                if len(_vals) >= 2 and min(_vals) > 0:
                    _spreads_pre.append(point_noisy(_vals))
    _invalid_pre = sum(_spreads_pre) > len(_spreads_pre) * args.max_noisy_fraction
    if _invalid_pre and os.path.exists(out):
        os.rename(out, out + ".stale")  # do not let an old file masquerade
    with open(out if not _invalid_pre else out + ".VOID-rc2", "w") as fh:
        fh.write("# Validated by validate_tuner_config.py: only rules that beat RCCL default\n")
        fh.write(f"# with P(sup) >= {args.min_psup} over {args.repeats} repeats, gain > {args.min_gain}%,\n")
        fh.write(f"# and no size in range regressing more than {args.max_regression}%.\n")
        for _n in scales:
            if pf_excluded.get(_n):
                fh.write(f"# preflight-excluded sizes at {_n}n (chronic same-config scatter over "
                         f"{args.max_arm_spread:.0f}%): {sorted(pf_excluded[_n])} -- "
                         f"no verdict was issued for these sizes.\n")
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
                if size in pf_excluded.get(nodes, set()):
                    continue  # scoped-preflight exclusion, see above
                if len(vals) >= 2 and min(vals) > 0:
                    spreads.append(((max(vals) - min(vals)) / min(vals) * 100,
                                    nodes, variant, size, point_noisy(vals)))
    worst_spread = max(spreads)[0] if spreads else 0.0
    noisy = [s for s in spreads if s[4]]

    print(f"\nwithin-arm spread: worst {worst_spread:.1f}%, "
          f"{len(noisy)} of {len(spreads)} (size, arm) points over {args.max_arm_spread:.0f}%")
    invalid = len(noisy) > len(spreads) * args.max_noisy_fraction

    print(f"\nkept {len(kept)} of {len(rules)} rules -> {out}")

    if invalid:
        wp, wn, wv, ws, _ = max(spreads)
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
