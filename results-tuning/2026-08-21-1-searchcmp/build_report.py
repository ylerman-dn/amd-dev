#!/usr/bin/env python3
"""Render searchcmp_summary.json + confs to report.md and report.html."""
import html as H
import json
from pathlib import Path

HERE = Path(__file__).parent
S = json.load(open(HERE / "searchcmp_summary.json"))
COLLS = ["all_reduce", "broadcast", "reduce", "all_gather", "reduce_scatter"]


def fmt_runs(v):
    return "-" if v is None else f"{v:.0f}"


def size_h(b):
    for u, d in (("M", 1 << 20), ("K", 1 << 10)):
        if b >= d:
            return f"{b // d}{u}"
    return str(b)


rows = []
for coll in COLLS:
    for n in (1, 2, 3):
        tag = f"{coll}_{n}n"
        r = S[tag]
        t = r["triage"]
        missed = t["worst_missed_pct"]
        rows.append({
            "grid_label": f"{coll} {n}n",
            "grid": r["grid"]["runs"] // 3,
            "adaptive": r["adaptive"]["runs"] // 3,
            "a_exact": f"{r['adaptive']['exact']}/{r['adaptive']['n_sizes']}",
            "a_gap": r["adaptive"]["worst_gap_pct"],
            "tpe": (r["tpe"]["runs_to_exact_median"] / 3
                    if r["tpe"]["runs_to_exact_median"] is not None else None),
            "tpe_found": f"{r['tpe']['found_exact']}/{r['tpe']['seeds']}",
            "rnd": (r["random"]["runs_to_exact_median"] / 3
                    if r["random"]["runs_to_exact_median"] is not None else None),
            "rnd_found": f"{r['random']['found_exact']}/{r['random']['seeds']}",
            "triage": t["runs"] // 3,
            "t_flag": len(t["flagged_tunable"]),
            "t_missed": missed,
            "flagged_h": ",".join(size_h(x) for x in t["flagged_tunable"]) or "none",
        })

tot = {k: sum(r[k] for r in rows) for k in ("grid", "adaptive", "triage")}

md = []
md.append("# Search-method comparison — 15 grids, 5 collectives × 1/2/3 nodes")
md.append("")
md.append("All methods replayed **offline against the same measured grid data** "
          "(no new cluster runs). Sources: all_reduce `2026-08-04-{1,4,6}`, "
          "broadcast/reduce `2026-08-18-1-newcolls`, all_gather/reduce_scatter "
          "`2026-08-19-2-agrs`.")
md.append("")
md.append("**Cost unit = measurements. 1 measurement = median of 3 "
          "benchmark runs** (the project's repeat policy; raw run counts = x3, "
          "kept in searchcmp_summary.json).")
md.append("")
md.append("**Scoring.** Grid = ground truth (its winners "
          "define 'exact'). Adaptive = racing search (anchors 1,8,24,48, "
          "margin 15, 3 repeats, tol 0.5). Optuna/random = runs until their "
          "winner set matches the grid's, median of 10 seeds ('found' = seeds "
          "that ever matched). Triage = 3 default runs + adaptive on dip-flagged "
          "sizes only (10% detector); 'missed' = biggest headroom it leaves at "
          "unflagged sizes.")
md.append("")
md.append("| grid | grid meas | adaptive | exact | gap% | optuna med | found | random med | found | triage | flagged | missed% |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    md.append(f"| {r['grid_label']} | {r['grid']} | {r['adaptive']} | "
              f"{r['a_exact']} | {r['a_gap']} | {fmt_runs(r['tpe'])} | "
              f"{r['tpe_found']} | {fmt_runs(r['rnd'])} | {r['rnd_found']} | "
              f"{r['triage']} | {r['flagged_h']} | {r['t_missed']} |")
md.append(f"| **total** | **{tot['grid']}** | **{tot['adaptive']}** | | | | | | | **{tot['triage']}** | | |")
md.append("")

md.append("## Configs (grid-truth pipeline: median → optimize -t 0.5 → tuner conf)")
for coll in COLLS:
    conf = (HERE / f"{coll}.conf").read_text().strip()
    md.append(f"\n### {coll}.conf\n```\n{conf}\n```")
md.append("\n### alltoall — no config")
md.append("RCCL forces RING/SIMPLE for alltoall at source (`enqueue.cc:2095`); "
          "algo/proto are not tunable and no grid was measured.")

md.append("\n## Caveats")
md.append("- broadcast/reduce 1n triage uses the only existing 1n default curve "
          "(2026-08-17-4-hotspot-mn, node 8) while their grids ran on node 6 - cross-node.")
md.append("- Each grid's node set differs (0804 vs {1,3}/{1,3,6} vs {5,7}/{5,6,7}); "
          "comparisons are within-grid only, never busbw across grids.")
md.append("- all_gather rows below 16M (1n) / 8M (2-3n) are Direct-substituted and "
          "excluded; the conf only covers the RING-honoured sizes, defaults rule below.")
md.append("- These configs are grid winners; only broadcast/reduce 2n/3n winners have "
          "been A/B-validated live (2026-08-19). The rest are unvalidated.")
md.append("- Optuna numbers use the findings/06 protocol; a real deployment cannot "
          "know when to stop — the medians shown are its *oracle-best* stopping point.")
md.append("- Measurement counts are runs/3, exact under the current repeat policy; "
          "if the 1+2 policy is ever used live, that table must show runs.")

(HERE / "report.md").write_text("\n".join(md) + "\n")

# ---- HTML (self-contained, phone-friendly)
css = """body{font-family:-apple-system,system-ui,sans-serif;margin:12px;max-width:900px}
table{border-collapse:collapse;font-size:13px;width:100%;overflow-x:auto;display:block}
th,td{border:1px solid #ccc;padding:4px 6px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
tr:nth-child(even){background:#f6f6f6}
pre{background:#f2f2f2;padding:8px;font-size:12px;overflow-x:auto;border-radius:6px}
h1{font-size:20px}h2{font-size:16px}h3{font-size:14px}
.best{background:#d7f5d7;font-weight:600}
.note{color:#555;font-size:13px}"""
h = [f"<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>"
     f"<title>GPU-107 search comparison</title><style>{css}</style>"]
h.append("<h1>Search-method comparison — 15 grids</h1>")
h.append("<p class=note>Offline replay on measured grid data, 2026-08-21. "
         "Cost unit = measurements; 1 measurement = median of 3 runs. Lower is better; 'exact' = same winners "
         "as the full grid.</p>")
h.append("<table><tr><th>grid</th><th>grid meas</th><th>adaptive</th><th>exact</th>"
         "<th>gap%</th><th>optuna</th><th>found</th><th>random</th><th>found</th>"
         "<th>triage</th><th>flagged sizes</th><th>missed%</th></tr>")
for r in rows:
    cand = {"adaptive": r["adaptive"],
            "optuna": r["tpe"] if r["tpe"] is not None else 10**9,
            "triage": r["triage"]}
    best = min(cand, key=cand.get)
    def td(key, val):
        cls = " class=best" if key == best else ""
        return f"<td{cls}>{val}</td>"
    h.append("<tr><td>" + r["grid_label"] + f"</td><td>{r['grid']}</td>" +
             td("adaptive", r["adaptive"]) + f"<td>{r['a_exact']}</td>"
             f"<td>{r['a_gap']}</td>" + td("optuna", fmt_runs(r["tpe"])) +
             f"<td>{r['tpe_found']}</td><td>{fmt_runs(r['rnd'])}</td>"
             f"<td>{r['rnd_found']}</td>" + td("triage", r["triage"]) +
             f"<td>{H.escape(r['flagged_h'])}</td><td>{r['t_missed']}</td></tr>")
h.append(f"<tr><td><b>total</b></td><td><b>{tot['grid']}</b></td>"
         f"<td><b>{tot['adaptive']}</b></td><td></td><td></td><td></td><td></td>"
         f"<td></td><td></td><td><b>{tot['triage']}</b></td><td></td><td></td></tr>")
h.append("</table>")
h.append("<p class=note>Green = cheapest method that still reaches the shown "
         "quality on that grid. Triage's 'missed%' is real performance left "
         "behind at sizes it chose not to search — its low cost is not free.</p>")
h.append("<h2>Per-collective tuner configs (grid truth)</h2>")
for coll in COLLS:
    conf = (HERE / f"{coll}.conf").read_text().strip()
    h.append(f"<h3>{coll}.conf</h3><pre>{H.escape(conf)}</pre>")
h.append("<h3>alltoall</h3><p class=note>No config: RCCL forces RING/SIMPLE at "
         "source; only channels would be tunable and no grid was measured.</p>")
h.append("<h2>Caveats</h2><ul class=note>"
         "<li>Node sets differ per grid — within-grid comparisons only.</li>"
         "<li>broadcast/reduce 1n triage curve is from node 8, their grids from node 6 (only 1n default curve available).</li>"
         "<li>all_gather below 16M/8M is Direct-substituted; conf covers RING sizes only.</li>"
         "<li>Only broadcast/reduce 2n/3n winners are live-A/B-validated; the rest are unvalidated grid winners.</li>"
         "<li>Optuna medians are its oracle-best stopping point (it cannot know when to stop in production).</li></ul>")
(HERE / "report.html").write_text("\n".join(h))
print("wrote report.md and report.html")
