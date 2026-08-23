#!/usr/bin/env python3
"""Adaptive (racing/prune) search over the RCCL sweep grid - GPU-107 follow-on.

Question: can we find the same per-size winners as the full grid sweep with
fewer benchmark runs?

Framing: one benchmark run returns all 18 message sizes at once, so a "trial"
scores 18 objectives. This is not one optimization problem but 18 argmax
problems sharing every evaluation. The search is therefore successive
elimination, not a sampler:

  Stage 1  anchors     - evaluate each algo/proto combo at a few channel values
  Stage 2  combo prune - drop combos beaten at EVERY size by more than a margin
                         (a combo useless at 4K may win at 512M, so ALL sizes)
  Stage 3  refinement  - hill-climb each survivor's channel curve per size
                         (neighbors of the per-size best), plus a downward walk
                         to serve optimize_metrics.py's min-channels tie-break
  Stage 4  selection   - the existing rule: max busbw_ip, tolerance band,
                         fewest channels among the tied (optimize_metrics.py)

Modes:
  selftest - prove the reimplemented selection rule reproduces optimized.csv
  replay   - simulate the search against an already-measured full grid
  tune     - grid over policy knobs x replay datasets, report savings/quality
  baseline - Optuna TPE / random search replayed on the same data, same metric
  live     - drive rccl_sweep.py for real over ssh (one command.txt per run)

Replay datasets:
  merged.csv from a grid sweep (per-repeat rows, e.g.
    results-tuning/2026-08-04-1-sweep1node/merged.csv), or
  variance.json (results-tuning/2026-08-16-12-variance/variance.json) with
    --nodes N in {1,2,3}.

Run counting: one "run" = one rccl-tests invocation = one (algo,proto,channels)
config x one repeat, exactly what the grid counts. Full-grid baseline for a
dataset = n_configs x 3.
"""

import argparse
import csv
import itertools
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

SIZES_18 = [4096 * (2 ** i) for i in range(18)]  # 4K .. 512M

# honoured algo/proto combos for all_reduce (findings/02, unsupported_combos.yaml)
COMBOS_1N = [("RING", "LL"), ("RING", "SIMPLE"), ("TREE", "LL")]
COMBOS_MN = [("RING", "LL"), ("RING", "LL128"), ("RING", "SIMPLE"),
             ("TREE", "LL"), ("TREE", "LL128"), ("TREE", "SIMPLE")]


# ---------------------------------------------------------------- datasets

def load_merged_csv(path):
    """merged.csv (per-repeat rows) -> data[(algo,proto,ch)][size] = [busbw,...]

    Honoured, non-default rows only (substituted == 0, requested_algo set).
    Keyed on the REQUESTED config - that is the knob the sweep turns.
    """
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("substituted") == "1":
                continue
            algo = (row.get("requested_algo") or "").upper()
            proto = (row.get("requested_proto") or "").upper()
            if not algo or not proto:
                continue  # default run, not part of the grid
            ch = int(float(row["requested_nchannels"]))
            size = int(row["size_bytes"])
            bw = float(row["busbw_ip"])
            data.setdefault((algo, proto, ch), {}).setdefault(size, []).append(bw)
    return data


def load_variance_json(path, nodes):
    """variance.json -> same structure. Row: [size, 'ALGO/PROTO/CH', r1, r2, r3,
    median, spread_pct, flag]. Rows labelled 'default' are excluded."""
    with open(path) as f:
        blob = json.load(f)
    rows = blob[str(nodes)]
    data = {}
    for r in rows:
        size, label = r[0], r[1]
        if label == "default" or "/" not in str(label):
            continue
        algo, proto, ch = label.split("/")
        reps = [float(x) for x in r[2:5]]
        med_file = float(r[5])
        med_calc = statistics.median(reps)
        if abs(med_file - med_calc) > 0.011:  # semantics check on the file
            raise ValueError(f"variance.json median mismatch at {label} {size}: "
                             f"file {med_file} vs computed {med_calc}")
        data[(algo.upper(), proto.upper(), int(ch))] = \
            {**data.get((algo.upper(), proto.upper(), int(ch)), {}),
             int(size): reps}
    # re-shape: ensure per-size lists
    out = {}
    for cfg, sizes in data.items():
        out[cfg] = {s: list(v) for s, v in sizes.items()}
    return out


def medians_of(data, only_cfgs=None):
    out = {}
    for cfg, sizes in data.items():
        if only_cfgs is not None and cfg not in only_cfgs:
            continue
        out[cfg] = {s: statistics.median(v) for s, v in sizes.items()}
    return out


# ------------------------------------------------- selection (optimize rule)

def select_winners(median_map, tol_pct):
    """Mirror of optimize_metrics.py: per size, max busbw_ip; configs within
    tol_pct of it are tied; fewest channels among the tied wins."""
    sizes = sorted({s for m in median_map.values() for s in m})
    winners = {}
    for size in sizes:
        avail = [(cfg, m[size]) for cfg, m in median_map.items() if size in m]
        best = max(bw for _, bw in avail)
        thr = best * (1 - tol_pct / 100.0)
        tied = [(cfg, bw) for cfg, bw in avail if bw >= thr]
        tied.sort(key=lambda t: (t[0][2], -t[1], t[0][0], t[0][1]))
        winners[size] = {"cfg": tied[0][0], "busbw": tied[0][1],
                         "best_busbw": best, "n_tied": len(tied)}
    return winners


# ------------------------------------------------------------------ oracles

class ReplayOracle:
    """Feeds stored repeats back one at a time. rotation shifts which stored
    repeat is consumed first (sensitivity for the 1-repeat explore policy)."""

    def __init__(self, data, rotation=0):
        self.data = data
        self.rotation = rotation
        self.consumed = {}   # cfg -> n repeats consumed
        self.runs = 0

    def available(self, cfg):
        return cfg in self.data

    def max_repeats(self, cfg):
        return min(len(v) for v in self.data[cfg].values())

    def run(self, cfg):
        i = self.consumed.get(cfg, 0)
        n = self.max_repeats(cfg)
        if i >= n:
            raise RuntimeError(f"no more stored repeats for {cfg}")
        idx = (i + self.rotation) % n
        self.consumed[cfg] = i + 1
        self.runs += 1
        return {s: v[idx] for s, v in self.data[cfg].items()}


# ------------------------------------------------------------- the policy

def adaptive_search(oracle, combos, grid, anchors, margin_pct,
                    explore_repeats, final_repeats, tol_pct, band_pct,
                    improve_eps_pct=2.0, sizes=None, trace=None):
    """Returns (winners, evals, stats). evals[cfg] = list of {size: busbw}.

    Refinement discipline (this is where the savings come from):
    - a combo is only refined at sizes where it is a CONTENDER (within
      margin_pct of the current leader at that size); a combo that wins only
      large sizes never has its small-size channel curve chased.
    - hill-climb with hysteresis: a newly evaluated channel only becomes the
      climb's incumbent if it beats it by > improve_eps_pct. Without this,
      noise on flat curves drags the climb across the whole ladder.
    - the min-channels tie-break of optimize_metrics.py is served by a
      downward walk: below the lowest in-band channel, one grid step at a
      time, while the new point still lands in the tolerance band.
    """
    grid = sorted(grid)
    trace = trace if trace is not None else []
    evals = {}
    eval_order = {}  # cfg -> index of first evaluation (hysteresis incumbent)

    def snap(v):
        return min(grid, key=lambda g: (abs(g - v), g))

    anchor_chs = sorted({snap(a) for a in anchors})

    def med(cfg, size):
        vals = [r[size] for r in evals[cfg] if size in r]
        return statistics.median(vals) if vals else None

    def run_cfg(cfg, repeats):
        if not oracle.available(cfg):
            trace.append(("unavailable", cfg))
            return False
        try:
            while len(evals.get(cfg, [])) < repeats:
                r = oracle.run(cfg)
                evals.setdefault(cfg, []).append(r)
                eval_order.setdefault(cfg, len(eval_order))
        except Exception as e:  # failed/substituted run: drop, keep going
            trace.append(("run_failed", cfg, str(e)))
            if not evals.get(cfg):
                evals.pop(cfg, None)
            return False
        return True

    # Stage 1: anchors
    for combo in combos:
        for ch in anchor_chs:
            run_cfg((combo[0], combo[1], ch), explore_repeats)
    live_combos = [c for c in combos
                   if any((c[0], c[1], ch) in evals for ch in anchor_chs)]
    if sizes is None:
        sizes = sorted({s for rs in evals.values() for r in rs for s in r})

    # Stage 2: prune combos dominated at EVERY size
    def combo_best(combo, size):
        vals = [med((combo[0], combo[1], ch), size) for ch in grid
                if (combo[0], combo[1], ch) in evals]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    overall = {s: max(combo_best(c, s) for c in live_combos
                      if combo_best(c, s) is not None) for s in sizes}
    alive = []
    for combo in live_combos:
        dominated = all(
            (combo_best(combo, s) or 0.0) < overall[s] * (1 - margin_pct / 100.0)
            for s in sizes)
        if dominated:
            trace.append(("pruned", combo,
                          {s: round((1 - (combo_best(combo, s) or 0) / overall[s]) * 100, 1)
                           for s in sizes}))
        else:
            alive.append(combo)

    def incumbent_ch(combo, size):
        """Argmax with hysteresis: a later-evaluated channel only displaces an
        earlier one if it wins by more than improve_eps_pct."""
        chs = [ch for ch in grid if (combo[0], combo[1], ch) in evals
               and med((combo[0], combo[1], ch), size) is not None]
        if not chs:
            return None
        chs.sort(key=lambda ch: eval_order.get((combo[0], combo[1], ch), 0))
        best = chs[0]
        for ch in chs[1:]:
            if med((combo[0], combo[1], ch), size) > \
                    med((combo[0], combo[1], best), size) * (1 + improve_eps_pct / 100.0):
                best = ch
        return best

    # Stage 3: contender-scoped hill-climb + tie-break downward walk
    while True:
        cands = set()
        for size in sizes:
            leader = max(m for m in (med(cfg, size) for cfg in evals)
                         if m is not None)
            band = leader * (1 - tol_pct / 100.0)
            contender_floor = leader * (1 - margin_pct / 100.0)
            for combo in alive:
                cb = combo_best(combo, size)
                if cb is None or cb < contender_floor:
                    continue  # not competitive at this size: don't refine here
                best_ch = incumbent_ch(combo, size)
                i = grid.index(best_ch)
                for j in (i - 1, i + 1):
                    if 0 <= j < len(grid):
                        cand = (combo[0], combo[1], grid[j])
                        if cand not in evals:
                            cands.add(cand)
                chs = [ch for ch in grid if (combo[0], combo[1], ch) in evals]
                in_band = [ch for ch in chs
                           if med((combo[0], combo[1], ch), size) >= band]
                if in_band:
                    i = grid.index(min(in_band))
                    if i > 0:
                        cand = (combo[0], combo[1], grid[i - 1])
                        if cand not in evals:
                            cands.add(cand)
        if not cands:
            break
        progressed = False
        for cfg in sorted(cands):
            if run_cfg(cfg, explore_repeats):
                progressed = True
        if not progressed:
            break

    # Stage 3b: finalists get topped up to final_repeats
    if explore_repeats < final_repeats:
        finalists = set()
        for size in sizes:
            leader = max(m for m in (med(cfg, size) for cfg in evals)
                         if m is not None)
            fin_band = leader * (1 - (tol_pct + band_pct) / 100.0)
            for cfg in evals:
                m = med(cfg, size)
                if m is not None and m >= fin_band:
                    finalists.add(cfg)
        for cfg in sorted(finalists):
            run_cfg(cfg, final_repeats)
        pool = {cfg: rs for cfg, rs in evals.items()
                if len(rs) >= final_repeats}
    else:
        pool = evals

    pool_medians = {cfg: {s: statistics.median([r[s] for r in rs if s in r])
                          for s in sizes if any(s in r for r in rs)}
                    for cfg, rs in pool.items()}
    winners = select_winners(pool_medians, tol_pct)
    stats = {"runs": oracle.runs if isinstance(oracle, ReplayOracle) else
             getattr(oracle, "runs", None),
             "configs": len(evals),
             "alive_combos": [list(c) for c in alive]}
    return winners, evals, stats


# ----------------------------------------------------------- replay report

def compare_winners(adaptive, grid_winners, full_medians):
    """Per size: exact match + bandwidth gap of the adaptive pick, measured on
    the FULL dataset medians (same data both sides)."""
    rows = []
    for size in sorted(grid_winners):
        gw = grid_winners[size]
        aw = adaptive.get(size)
        a_cfg = aw["cfg"] if aw else None
        a_bw_full = full_medians.get(a_cfg, {}).get(size) if a_cfg else None
        gap = None
        if a_bw_full is not None and gw["busbw"]:
            gap = (gw["busbw"] - a_bw_full) / gw["busbw"] * 100.0
        rows.append({
            "size": size,
            "grid_cfg": "/".join(map(str, gw["cfg"])),
            "grid_busbw": gw["busbw"],
            "adaptive_cfg": "/".join(map(str, a_cfg)) if a_cfg else None,
            "adaptive_busbw_fullmed": a_bw_full,
            "exact": a_cfg == gw["cfg"],
            "gap_pct": round(gap, 3) if gap is not None else None,
            "ch_delta": (a_cfg[2] - gw["cfg"][2]) if a_cfg else None,
        })
    return rows


def run_replay(data, combos, anchors, margin, repeat_policy, band, tol,
               rotation=0):
    grid = sorted({cfg[2] for cfg in data})
    explore, final = (3, 3) if repeat_policy == "3" else (1, 3)
    oracle = ReplayOracle(data, rotation=rotation)
    winners, evals, stats = adaptive_search(
        oracle, combos, grid, anchors, margin, explore, final, tol, band)
    full_medians = medians_of(data)
    grid_winners = select_winners(full_medians, tol)
    rows = compare_winners(winners, grid_winners, full_medians)
    full_runs = len(data) * 3
    return {
        "runs": oracle.runs, "full_runs": full_runs,
        "saving_pct": round((1 - oracle.runs / full_runs) * 100, 1),
        "configs": stats["configs"], "full_configs": len(data),
        "alive_combos": stats["alive_combos"],
        "exact": sum(r["exact"] for r in rows),
        "n_sizes": len(rows),
        "worst_gap_pct": max((r["gap_pct"] for r in rows
                              if r["gap_pct"] is not None), default=None),
        "rows": rows,
    }


# ------------------------------------------------------------ optuna/random

def scalarize(median_map, size_max):
    """Replay-only scalar objective: mean over sizes of busbw / per-size max."""
    def f(cfg):
        m = median_map[cfg]
        return statistics.mean(m[s] / size_max[s] for s in size_max if s in m)
    return f


def run_sampler_baseline(data, sampler_name, tol, seed, gap_ok=5.0):
    """How many unique configs must a sampler evaluate before the winners
    selected from its evaluated set match the full grid?  Duplicate asks are
    free (cached). Runs = unique configs x 3 (same repeats as the grid)."""
    full_medians = medians_of(data)
    grid_winners = select_winners(full_medians, tol)
    sizes = sorted({s for m in full_medians.values() for s in m})
    size_max = {s: max(m[s] for m in full_medians.values() if s in m)
                for s in sizes}
    obj = scalarize(full_medians, size_max)
    cfgs = sorted(data)
    combos = sorted({(c[0], c[1]) for c in cfgs})
    chans = sorted({c[2] for c in cfgs})

    order = []
    if sampler_name == "random":
        rng = random.Random(seed)
        order = cfgs[:]
        rng.shuffle(order)
    elif sampler_name == "tpe":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        seen = set()
        # cap asks: TPE re-asks duplicates; cached, but bound the loop
        asks = 0
        while len(seen) < len(cfgs) and asks < len(cfgs) * 30:
            trial = study.ask()
            combo = trial.suggest_categorical("combo",
                                              ["/".join(c) for c in combos])
            ch = trial.suggest_categorical("ch", chans)
            algo, proto = combo.split("/")
            cfg = (algo, proto, ch)
            asks += 1
            if cfg not in data:
                study.tell(trial, 0.0)  # not honoured at this scale
                continue
            study.tell(trial, obj(cfg))
            if cfg not in seen:
                seen.add(cfg)
                order.append(cfg)
    else:
        raise ValueError(sampler_name)

    evaluated = {}
    first_exact = None
    first_gap_ok = None
    for i, cfg in enumerate(order, 1):
        evaluated[cfg] = full_medians[cfg]
        w = select_winners(evaluated, tol)
        if first_exact is None and all(
                w[s]["cfg"] == grid_winners[s]["cfg"] for s in sizes):
            first_exact = i
        if first_gap_ok is None:
            ok = True
            for s in sizes:
                pick = w[s]["cfg"]
                gap = (grid_winners[s]["busbw"] - full_medians[pick][s]) \
                    / grid_winners[s]["busbw"] * 100.0
                if gap > gap_ok:
                    ok = False
                    break
            if ok:
                first_gap_ok = i
    return {"sampler": sampler_name, "seed": seed,
            "configs_total": len(cfgs),
            "configs_to_exact": first_exact,
            "configs_to_gap5": first_gap_ok,
            "runs_to_exact": first_exact * 3 if first_exact else None,
            "runs_to_gap5": first_gap_ok * 3 if first_gap_ok else None}


# ------------------------------------------------------------------- live

class LiveOracle:
    """Runs rccl_sweep.py on a cluster node over ssh, one config per call.
    Every invocation gets its own --output-dir (r0000, r0001, ...) so each
    benchmark run keeps its own command.txt / output.log / dbg logs."""

    def __init__(self, exec_node, remote_tool, remote_out, servers_file,
                 nodes, collective, min_size, max_size, my_path, log):
        self.exec_node = exec_node
        self.remote_tool = remote_tool
        self.remote_out = remote_out
        self.servers_file = servers_file
        self.nodes = nodes
        self.collective = collective
        self.min_size = min_size
        self.max_size = max_size
        self.my_path = my_path
        self.log = log
        self.counter = 0
        self.runs = 0
        self.failures = []

    def available(self, cfg):
        return True  # live: substitution is detected after the run

    def run(self, cfg):
        algo, proto, ch = cfg
        rid = f"r{self.counter:04d}"
        self.counter += 1
        out = f"{self.remote_out}/{rid}"
        cmd = (f"cd {self.remote_tool} && export MY_PATH={self.my_path} && "
               f"python3 rccl_sweep.py --servers {self.servers_file} "
               f"--output-dir {out} --nodes {self.nodes} "
               f"--collective {self.collective} "
               f"--min-size {self.min_size} --max-size {self.max_size} "
               f"--channels {ch} --algo {algo} --proto {proto}")
        t0 = time.time()
        p = subprocess.run(["ssh", self.exec_node, cmd],
                           stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, timeout=900)
        dt = time.time() - t0
        self.runs += 1
        metrics = subprocess.run(
            ["ssh", self.exec_node, f"cat {out}/run_*/metrics.csv"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True)
        with open(self.log, "a") as f:
            f.write(f"{rid} cfg={algo}/{proto}/{ch} rc={p.returncode} "
                    f"dur={dt:.0f}s out={out}\n")
        if p.returncode != 0 or metrics.returncode != 0:
            self.failures.append((rid, cfg, p.returncode))
            raise RuntimeError(f"live run {rid} {cfg} failed rc={p.returncode}: "
                               f"{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
        result = {}
        subst = 0
        for row in csv.DictReader(metrics.stdout.splitlines()):
            if row.get("substituted") == "1":
                subst += 1
            result[int(row["size_bytes"])] = float(row["busbw_ip"])
        if subst:
            raise SubstitutedError(cfg, subst, len(result))
        if not result:
            raise RuntimeError(f"live run {rid} {cfg}: empty metrics")
        return result


class SubstitutedError(RuntimeError):
    def __init__(self, cfg, n, total):
        super().__init__(f"{cfg} substituted at {n}/{total} sizes")
        self.cfg = cfg


# --------------------------------------------------------------- commands

def cmd_selftest(args):
    """Prove select_winners reproduces the committed optimized.csv (tol=5.0,
    the default at the time it was generated)."""
    base = Path(args.dataset).parent
    med_map = {}
    with open(base / "merged_median.csv") as f:
        for row in csv.DictReader(f):
            algo = (row.get("requested_algo") or "").upper()
            proto = (row.get("requested_proto") or "").upper()
            if not algo or not proto or row.get("substituted") == "1":
                continue
            cfg = (algo, proto, int(float(row["nchannels"])))
            med_map.setdefault(cfg, {})[int(row["size_bytes"])] = \
                float(row["busbw_ip"])
    winners = select_winners(med_map, 5.0)
    ok = bad = 0
    with open(base / "optimized.csv") as f:
        for row in csv.DictReader(f):
            size = int(row["size_bytes"])
            exp = (row["requested_algo"].upper(), row["requested_proto"].upper(),
                   int(float(row["nchannels"])))
            got = winners[size]["cfg"]
            if got == exp:
                ok += 1
            else:
                bad += 1
                print(f"  MISMATCH {size}: mine {got} vs optimized.csv {exp} "
                      f"(mine bw {winners[size]['busbw']}, n_tied "
                      f"{winners[size]['n_tied']})")
    print(f"selftest vs {base}/optimized.csv: {ok} match, {bad} mismatch")
    return 0 if bad == 0 else 1


def load_dataset(args):
    if args.dataset.endswith(".json"):
        if not args.nodes:
            sys.exit("--nodes required with variance.json")
        return load_variance_json(args.dataset, args.nodes)
    return load_merged_csv(args.dataset)


def combos_for(data):
    return sorted({(c[0], c[1]) for c in data})


def cmd_replay(args):
    data = load_dataset(args)
    rep = run_replay(data, combos_for(data),
                     [int(a) for a in args.anchors.split(",")],
                     args.margin, args.repeat_policy, args.band, args.tol,
                     rotation=args.rotation)
    print(json.dumps(rep, indent=2))
    return 0


def cmd_tune(args):
    datasets = []
    for spec in args.datasets.split():
        if ":" in spec and spec.endswith((":1", ":2", ":3")):
            path, n = spec.rsplit(":", 1)
            datasets.append((f"{Path(path).parent.name}:{n}n",
                             load_variance_json(path, int(n))))
        else:
            datasets.append((Path(spec).parent.name, load_merged_csv(spec)))
    anchor_sets = [tuple(int(x) for x in a.split(","))
                   for a in args.anchor_sets.split()]
    margins = [float(m) for m in args.margins.split(",")]
    policies = args.repeat_policies.split(",")
    results = []
    for anchors, margin, policy in itertools.product(
            anchor_sets, margins, policies):
        agg = {"anchors": anchors, "margin": margin, "policy": policy,
               "per_dataset": {}}
        for name, data in datasets:
            rotations = [0, 1, 2] if policy != "3" else [0]
            worst = None
            for rot in rotations:
                rep = run_replay(data, combos_for(data), anchors, margin,
                                 policy, args.band, args.tol, rotation=rot)
                key = (rep["exact"], -(rep["worst_gap_pct"] or 0),
                       rep["saving_pct"])
                if worst is None or key < worst[0]:
                    worst = (key, rep)
            rep = worst[1]
            agg["per_dataset"][name] = {
                "saving_pct": rep["saving_pct"], "runs": rep["runs"],
                "full_runs": rep["full_runs"], "exact": rep["exact"],
                "n_sizes": rep["n_sizes"],
                "worst_gap_pct": rep["worst_gap_pct"]}
        ds = agg["per_dataset"].values()
        agg["min_saving"] = min(d["saving_pct"] for d in ds)
        agg["min_exact"] = min(d["exact"] for d in ds)
        agg["max_gap"] = max((d["worst_gap_pct"] or 0) for d in ds)
        agg["gate"] = (agg["min_saving"] >= args.gate_saving
                       and agg["max_gap"] <= args.gate_gap)
        results.append(agg)
    results.sort(key=lambda a: (-a["gate"], -a["min_exact"], a["max_gap"],
                                -a["min_saving"]))
    print(json.dumps(results, indent=2, default=list))
    return 0


def cmd_baseline(args):
    data = load_dataset(args)
    out = []
    for sampler in args.samplers.split(","):
        for seed in range(args.seeds):
            out.append(run_sampler_baseline(data, sampler, args.tol, seed))
    print(json.dumps(out, indent=2))
    return 0


def cmd_live(args):
    combos = COMBOS_1N if args.nodes == 1 else COMBOS_MN
    grid = [int(x) for x in args.grid.split(",")]
    anchors = [int(x) for x in args.anchors.split(",")]
    explore, final = (3, 3) if args.repeat_policy == "3" else (1, 3)
    Path(args.local_out).mkdir(parents=True, exist_ok=True)
    log = Path(args.local_out) / "live_runs.log"
    oracle = LiveOracle(args.exec_node, args.remote_tool, args.remote_out,
                        args.servers_file, args.nodes, args.collective,
                        args.min_size, args.max_size, args.my_path, log)

    # substitution-aware wrapper: a substituted combo is dropped whole
    class Guard:
        def __init__(self, inner):
            self.inner = inner
            self.dead_combos = set()
            self.runs = 0

        def available(self, cfg):
            return (cfg[0], cfg[1]) not in self.dead_combos

        def run(self, cfg):
            try:
                r = self.inner.run(cfg)
                self.runs = self.inner.runs
                return r
            except SubstitutedError as e:
                self.dead_combos.add((cfg[0], cfg[1]))
                self.runs = self.inner.runs
                raise RuntimeError(str(e))

    guard = Guard(oracle)
    trace = []
    t0 = time.time()
    winners, evals, stats = adaptive_search(
        guard, combos, grid, anchors, args.margin, explore, final,
        args.tol, args.band, sizes=None, trace=trace)
    search_wall_s = time.time() - t0

    # default (NCCL-auto) runs for same-day context, like the grid's 3
    default_bw = []
    for i in range(args.default_runs):
        rid = f"d{i:04d}"
        out = f"{args.remote_out}/{rid}"
        cmd = (f"cd {args.remote_tool} && export MY_PATH={args.my_path} && "
               f"python3 rccl_sweep.py --servers {args.servers_file} "
               f"--output-dir {out} --nodes {args.nodes} "
               f"--collective {args.collective} "
               f"--min-size {args.min_size} --max-size {args.max_size}")
        p = subprocess.run(["ssh", args.exec_node, cmd],
                           stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, timeout=900)
        m = subprocess.run(["ssh", args.exec_node,
                            f"cat {out}/run_*/metrics.csv"],
                           stdin=subprocess.DEVNULL, capture_output=True,
                           text=True)
        with open(log, "a") as f:
            f.write(f"{rid} cfg=default rc={p.returncode} out={out}\n")
        if p.returncode == 0 and m.returncode == 0:
            default_bw.append({int(r["size_bytes"]): float(r["busbw_ip"])
                               for r in csv.DictReader(m.stdout.splitlines())})

    report = {
        "policy": {"anchors": anchors, "margin": args.margin,
                   "repeat_policy": args.repeat_policy, "tol": args.tol,
                   "band": args.band, "grid": grid,
                   "combos": [list(c) for c in combos]},
        "winners": {str(s): {"cfg": "/".join(map(str, w["cfg"])),
                             "busbw": w["busbw"]}
                    for s, w in winners.items()},
        "runs": oracle.runs, "configs": stats["configs"],
        "default_runs": len(default_bw),
        "search_wall_s": round(search_wall_s, 1),
        "alive_combos": stats["alive_combos"],
        "dead_combos": [list(c) for c in guard.dead_combos],
        "failures": [(r, list(c), rc) for r, c, rc in oracle.failures],
        "trace": [str(t) for t in trace],
        "default_busbw": [{str(s): v for s, v in d.items()}
                          for d in default_bw],
        "evals": {"/".join(map(str, cfg)):
                  {str(s): [r.get(s) for r in rs] for s in sorted(rs[0])}
                  for cfg, rs in evals.items()},
    }
    out = Path(args.local_out) / "adaptive_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("winners", "runs", "configs", "alive_combos",
                       "dead_combos", "failures")}, indent=2))
    print(f"report: {out}")
    return 0


def cmd_compare(args):
    """Live adaptive report vs a reference full grid: same winners? and the
    bandwidth gap, measured in BOTH sessions' own data where possible."""
    with open(args.report) as f:
        rep = json.load(f)
    live_evals = {}
    for cfg_s, sizes in rep["evals"].items():
        algo, proto, ch = cfg_s.split("/")
        live_evals[(algo, proto, int(ch))] = \
            {int(s): [v for v in vals if v is not None]
             for s, vals in sizes.items()}
    live_medians = {cfg: {s: statistics.median(v) for s, v in m.items() if v}
                    for cfg, m in live_evals.items()}
    ref = load_dataset(args)
    ref_medians = medians_of(ref)
    ref_winners = select_winners(ref_medians, args.tol)
    out = []
    for s_str, w in sorted(rep["winners"].items(), key=lambda kv: int(kv[0])):
        size = int(s_str)
        a_cfg = tuple(w["cfg"].split("/")[:2]) + (int(w["cfg"].split("/")[2]),)
        r = ref_winners.get(size)
        r_cfg = r["cfg"] if r else None
        # gap in the LIVE session: live median of ref winner vs live pick
        live_gap = None
        if r_cfg in live_medians and size in live_medians[r_cfg] \
                and size in live_medians.get(a_cfg, {}):
            ref_in_live = live_medians[r_cfg][size]
            pick_in_live = live_medians[a_cfg][size]
            live_gap = (ref_in_live - pick_in_live) / ref_in_live * 100.0
        # gap in the REFERENCE grid: ref median of live pick vs ref winner
        ref_gap = None
        if a_cfg in ref_medians and size in ref_medians[a_cfg] and r:
            ref_gap = (r["busbw"] - ref_medians[a_cfg][size]) \
                / r["busbw"] * 100.0
        out.append({
            "size": size,
            "live_cfg": w["cfg"], "live_busbw": w["busbw"],
            "ref_cfg": "/".join(map(str, r_cfg)) if r_cfg else None,
            "ref_busbw": r["busbw"] if r else None,
            "exact": (r_cfg == a_cfg),
            "gap_pct_in_live_data": round(live_gap, 3)
            if live_gap is not None else None,
            "gap_pct_in_ref_data": round(ref_gap, 3)
            if ref_gap is not None else None,
            "ref_winner_evaluated_live": r_cfg in live_medians,
        })
    summary = {
        "exact": sum(r["exact"] for r in out), "n_sizes": len(out),
        "worst_gap_in_live_data": max((r["gap_pct_in_live_data"] for r in out
                                       if r["gap_pct_in_live_data"] is not None),
                                      default=None),
        "worst_gap_in_ref_data": max((r["gap_pct_in_ref_data"] for r in out
                                      if r["gap_pct_in_ref_data"] is not None),
                                     default=None),
        "rows": out,
    }
    print(json.dumps(summary, indent=2))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tol", type=float, default=0.5,
                        help="selection tolerance %% (optimize_metrics.py rule)")
    common.add_argument("--anchors", default="8,48")
    common.add_argument("--margin", type=float, default=15.0,
                        help="stage-2 domination margin %%")
    common.add_argument("--repeat-policy", choices=["3", "1+2"], default="3")
    common.add_argument("--band", type=float, default=3.0,
                        help="finalist window %% beyond tol (1+2 policy)")

    ps = sub.add_parser("selftest", parents=[common])
    ps.add_argument("--dataset", required=True,
                    help="merged.csv path (its dir must hold merged_median.csv"
                         " and optimized.csv)")

    pr = sub.add_parser("replay", parents=[common])
    pr.add_argument("--dataset", required=True)
    pr.add_argument("--nodes", type=int)
    pr.add_argument("--rotation", type=int, default=0)

    pt = sub.add_parser("tune", parents=[common])
    pt.add_argument("--datasets", required=True,
                    help="space-separated: merged.csv paths and "
                         "variance.json:N specs")
    pt.add_argument("--anchor-sets", required=True,
                    help="space-separated comma-lists, e.g. '8,48 4,32'")
    pt.add_argument("--margins", default="10,15,20,25")
    pt.add_argument("--repeat-policies", default="3,1+2")
    pt.add_argument("--gate-saving", type=float, default=30.0)
    pt.add_argument("--gate-gap", type=float, default=5.0)

    pb = sub.add_parser("baseline", parents=[common])
    pb.add_argument("--dataset", required=True)
    pb.add_argument("--nodes", type=int)
    pb.add_argument("--samplers", default="tpe,random")
    pb.add_argument("--seeds", type=int, default=10)

    pl = sub.add_parser("live", parents=[common])
    pl.add_argument("--nodes", type=int, required=True)
    pl.add_argument("--grid", required=True,
                    help="channel grid, e.g. 1,2,4,8,16,24,32,40,48,56")
    pl.add_argument("--exec-node", default="amd-mi355x-7")
    pl.add_argument("--remote-tool", required=True)
    pl.add_argument("--remote-out", required=True)
    pl.add_argument("--servers-file", required=True)
    pl.add_argument("--local-out", required=True)
    pl.add_argument("--collective", default="all_reduce")
    pl.add_argument("--min-size", default="4K")
    pl.add_argument("--max-size", default="512M")
    pl.add_argument("--my-path", default="/opt/shared/ylerman/GPU-107/bin")
    pl.add_argument("--default-runs", type=int, default=3,
                    help="NCCL-default context runs after the search")

    pc = sub.add_parser("compare", parents=[common])
    pc.add_argument("--report", required=True,
                    help="adaptive_report.json from a live run")
    pc.add_argument("--dataset", required=True,
                    help="reference full grid (merged.csv or variance.json)")
    pc.add_argument("--nodes", type=int)

    args = p.parse_args()
    return {"selftest": cmd_selftest, "replay": cmd_replay, "tune": cmd_tune,
            "baseline": cmd_baseline, "live": cmd_live}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
