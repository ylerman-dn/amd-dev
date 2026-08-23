#!/usr/bin/env python3
"""Offline search-method comparison on the 15 measured grids.

Methods, all replayed against the SAME grid data (no cluster time):
  grid     - full grid = ground truth; cost = all runs
  adaptive - racing search, policy from findings/07 (anchors 1,8,24,48,
             margin 15, repeat-policy 3, tol 0.5)
  optuna   - TPE sampler, 10 seeds, runs-to-exact protocol of findings/06
  random   - random order sampler, same protocol (optuna's control)
  triage   - dip detector (10%) on the default curve picks the sizes worth
             searching; adaptive runs only there; unflagged sizes keep the
             RCCL default. Cost includes the 3 default runs it needs.

Outputs per grid: replay_<coll>_<n>n.json, optuna_<coll>_<n>n.json,
<coll>_<n>n_{median,optimized}.csv, confs, and searchcmp_summary.json.
"""
import csv, json, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
TOOLS = Path("/home/dn/ylerman/tasks/GPU-107/amd-dev/tools/rccl-sweep")
RT = Path("/home/dn/ylerman/tasks/GPU-107/amd-dev-optuna/results-tuning")
PY = "/usr/bin/python3"
sys.path.insert(0, str(TOOLS))
import adaptive_search as A

GRIDS = {
    ("all_reduce", 1): RT / "2026-08-04-1-sweep1node/merged.csv",
    ("all_reduce", 2): RT / "2026-08-04-4-sweep2node/merged.csv",
    ("all_reduce", 3): RT / "2026-08-04-6-sweep3node/merged.csv",
    ("broadcast", 1): RT / "2026-08-18-1-newcolls/broadcast_1n_merged.csv",
    ("broadcast", 2): RT / "2026-08-18-1-newcolls/broadcast_2n_merged.csv",
    ("broadcast", 3): RT / "2026-08-18-1-newcolls/broadcast_3n_merged.csv",
    ("reduce", 1): RT / "2026-08-18-1-newcolls/reduce_1n_merged.csv",
    ("reduce", 2): RT / "2026-08-18-1-newcolls/reduce_2n_merged.csv",
    ("reduce", 3): RT / "2026-08-18-1-newcolls/reduce_3n_merged.csv",
    ("all_gather", 1): RT / "2026-08-19-2-agrs/all_gather_1n_merged.csv",
    ("all_gather", 2): RT / "2026-08-19-2-agrs/all_gather_2n_merged.csv",
    ("all_gather", 3): RT / "2026-08-19-2-agrs/all_gather_3n_merged.csv",
    ("reduce_scatter", 1): RT / "2026-08-19-2-agrs/reduce_scatter_1n_merged.csv",
    ("reduce_scatter", 2): RT / "2026-08-19-2-agrs/reduce_scatter_2n_merged.csv",
    ("reduce_scatter", 3): RT / "2026-08-19-2-agrs/reduce_scatter_3n_merged.csv",
}
DEFAULT_CURVES = {  # collective -> csv with DEFAULT rows (same-session data)
    "broadcast": RT / "2026-08-18-1-newcolls/newcolls_default_curves.csv",
    "reduce": RT / "2026-08-18-1-newcolls/newcolls_default_curves.csv",
    "all_gather": RT / "2026-08-19-2-agrs/agrs_default_curves.csv",
    "reduce_scatter": RT / "2026-08-19-2-agrs/agrs_default_curves.csv",
}
ANCHORS, MARGIN, POLICY, TOL, BAND = [1, 8, 24, 48], 15.0, "3", 0.5, 3.0
DIP_THRESHOLD = 0.10


def sh(*cmd):
    r = subprocess.run(list(cmd), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def default_curve(coll, nodes):
    """size -> median default busbw, from same-session default measurements."""
    out = {}
    if coll == "all_reduce":  # defaults live inside the 0804 grids
        with open(GRIDS[(coll, nodes)]) as f:
            for row in csv.DictReader(f):
                if (row.get("requested_algo") or "").strip():
                    continue
                out.setdefault(int(row["size_bytes"]), []).append(
                    float(row["busbw_ip"]))
        return {s: statistics.median(v) for s, v in out.items()}
    # newcolls default curves lack 1n; the only 1n broadcast/reduce default
    # curve is 2026-08-17-4-hotspot-mn (node 8, NOT the grid's node 6 -
    # cross-node, flagged in the report)
    sources = [DEFAULT_CURVES[coll],
               RT / "2026-08-17-4-hotspot-mn/hotspot_mn_curves.csv"]
    for src in sources:
        with open(src) as f:
            for row in csv.DictReader(f):
                if row["collective"] != f"{coll}_perf" or \
                   int(row["num_nodes"]) != nodes:
                    continue
                out.setdefault(int(row["size_bytes"]), []).append(
                    float(row["busbw_ip"]))
        if out:
            break
    return {s: statistics.median(v) for s, v in out.items()}


def detect_dips(curve, threshold=DIP_THRESHOLD):
    """detect_hotspots.py rule: flag sizes below running-max * (1-threshold)."""
    flagged, running = [], 0.0
    for s in sorted(curve):
        if running and curve[s] < running * (1 - threshold):
            flagged.append(s)
        running = max(running, curve[s])
    return flagged


def main():
    summary = {}
    for (coll, nodes), grid_csv in sorted(GRIDS.items()):
        tag = f"{coll}_{nodes}n"
        print(f"=== {tag} ===", flush=True)
        res = {"grid_csv": str(grid_csv)}

        # ---- grid truth: median -> optimize -> conf (the shipped pipeline)
        med = HERE / f"{tag}_median.csv"
        opt = HERE / f"{tag}_optimized.csv"
        if not (HERE / f"{tag}_grid.conf").exists():
            sh(PY, str(TOOLS / "merge_metrics.py"), "--input", str(grid_csv),
               "--median", "--output", str(med))
            sh(PY, str(TOOLS / "optimize_metrics.py"), str(med), "-o", str(opt),
               "-t", str(TOL))
            sh(PY, str(TOOLS / "generate_tuner_config.py"), str(opt),
               "-o", str(HERE / f"{tag}_grid.conf"), "--include-algo-proto")

        data = A.load_merged_csv(str(grid_csv))
        full_med = A.medians_of(data)
        grid_w = A.select_winners(full_med, TOL)
        res["grid"] = {"runs": len(data) * 3, "configs": len(data),
                       "sizes": len(grid_w)}

        # ---- adaptive replay + emitted conf
        rep = A.run_replay(data, A.combos_for(data), ANCHORS, MARGIN,
                           POLICY, BAND, TOL)
        (HERE / f"replay_{tag}.json").write_text(json.dumps(rep, indent=2))
        aopt = HERE / f"{tag}_adaptive_optimized.csv"
        A.emit_optimized_csv(
            [(r["size"], r["adaptive_cfg"], r["adaptive_busbw_fullmed"])
             for r in rep["rows"]],
            A.merged_csv_context(str(grid_csv)), aopt)
        sh(PY, str(TOOLS / "generate_tuner_config.py"), str(aopt),
           "-o", str(HERE / f"{tag}_adaptive.conf"), "--include-algo-proto")
        same_conf = (HERE / f"{tag}_grid.conf").read_text() == \
                    (HERE / f"{tag}_adaptive.conf").read_text()
        res["adaptive"] = {"runs": rep["runs"], "exact": rep["exact"],
                           "n_sizes": rep["n_sizes"],
                           "worst_gap_pct": rep["worst_gap_pct"],
                           "saving_pct": rep["saving_pct"],
                           "conf_identical_to_grid": same_conf}

        # ---- optuna + random baselines (findings/06 protocol)
        oj = HERE / f"optuna_{tag}.json"
        if oj.exists():
            bl = json.loads(oj.read_text())
        else:
            bl = json.loads(sh(PY, str(TOOLS / "adaptive_search.py"), "baseline",
                               "--dataset", str(grid_csv),
                               "--samplers", "tpe,random", "--seeds", "10",
                               "--tol", str(TOL)))
            oj.write_text(json.dumps(bl, indent=2))
        for name in ("tpe", "random"):
            rs = [b["runs_to_exact"] for b in bl if b["sampler"] == name]
            found = [r for r in rs if r is not None]
            res[name] = {
                "seeds": len(rs), "found_exact": len(found),
                "runs_to_exact_median": statistics.median(found) if found else None,
                "runs_to_exact_min": min(found) if found else None,
                "runs_to_exact_max": max(found) if found else None}

        # ---- dip triage simulation
        curve = default_curve(coll, nodes)
        flagged = detect_dips(curve)
        tunable = sorted({s for m in full_med.values() for s in m})
        flagged_t = [s for s in flagged if s in tunable]
        if flagged_t:
            fdata = {c: {s: v for s, v in m.items() if s in flagged_t}
                     for c, m in data.items()}
            fdata = {c: m for c, m in fdata.items() if m}
            frep = A.run_replay(fdata, A.combos_for(fdata), ANCHORS, MARGIN,
                                POLICY, BAND, TOL)
            triage_runs = 3 + frep["runs"]  # 3 default runs buy the curve
        else:
            triage_runs = 3
        # what triage leaves on the table: headroom at unflagged sizes
        missed = []
        for s in tunable:
            if s in flagged_t or s not in curve:
                continue
            gw = grid_w[s]
            head = (gw["busbw"] - curve[s]) / gw["busbw"] * 100.0
            if head > 0:
                missed.append((s, round(head, 1)))
        res["triage"] = {
            "default_curve_sizes": len(curve),
            "flagged": flagged, "flagged_tunable": flagged_t,
            "runs": triage_runs,
            "missed_headroom_pct": dict(missed),
            "worst_missed_pct": max((h for _, h in missed), default=0.0)}
        summary[tag] = res
        print(json.dumps({k: res[k] for k in
                          ("grid", "adaptive", "tpe", "random", "triage")},
                         indent=None), flush=True)

    # ---- per-collective configs (scales concatenated; generator groups by
    # (collective, nNodes, nRanks) so ranges never merge across scales)
    for coll in sorted({c for c, _ in GRIDS}):
        rows, fields = [], None
        for nodes in (1, 2, 3):
            with open(HERE / f"{coll}_{nodes}n_optimized.csv") as f:
                r = csv.DictReader(f)
                fields = fields or r.fieldnames
                rows += list(r)
        cat = HERE / f"{coll}_all_scales_optimized.csv"
        with open(cat, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        sh(PY, str(TOOLS / "generate_tuner_config.py"), str(cat),
           "-o", str(HERE / f"{coll}.conf"), "--include-algo-proto")
        print(f"config: {coll}.conf", flush=True)

    (HERE / "searchcmp_summary.json").write_text(json.dumps(summary, indent=2))
    print("summary: searchcmp_summary.json")


if __name__ == "__main__":
    main()
