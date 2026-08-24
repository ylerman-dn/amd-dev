#!/usr/bin/env python3
"""Build the GPU-107 presentation: index + per-collective gains pages.

Gains = our config's winner vs RCCL default, per size, per scale.
Every number is labeled:
  predicted  - from the grid measurements (env-var forcing path)
  validated  - live A/B through the tuner plugin (validate_tuner_config.py)
Only broadcast 3n, reduce 2n, reduce 3n have live verdicts (2026-08-19).
"""
import csv, html, json, re, statistics
from pathlib import Path

HERE = Path(__file__).parent
RT = HERE.parent
CMP = RT / "2026-08-21-1-searchcmp"
COLLS = ["all_reduce", "broadcast", "reduce", "all_gather", "reduce_scatter"]
AB = {  # (coll, nodes) -> validated.csv with live verdicts
    ("broadcast", 3): "2026-08-19-1-bcmn-ab/broadcast_3n.validated.csv",
    ("reduce", 2): "2026-08-19-1-bcmn-ab/reduce_2n.validated.csv",
    ("reduce", 3): "2026-08-19-1-bcmn-ab/reduce_3n.validated.csv",
    # 2026-08-23 batch on {5,8}, job 20713
    ("all_reduce", 1): "2026-08-23-2-abvalidate/all_reduce_1n_grid.validated.csv",
    ("all_reduce", 2): "2026-08-23-2-abvalidate/all_reduce_2n_grid.validated.csv",
    ("broadcast", 1): "2026-08-23-2-abvalidate/broadcast_1n_grid.validated.csv",
    ("reduce", 1): "2026-08-23-2-abvalidate/reduce_1n_grid.validated.csv",
    ("all_gather", 1): "2026-08-23-2-abvalidate/all_gather_1n_grid.validated.csv",
    ("all_gather", 2): "2026-08-23-2-abvalidate/all_gather_2n_grid.validated.csv",
    ("reduce_scatter", 1): "2026-08-23-2-abvalidate/reduce_scatter_1n_grid.validated.csv",
    ("reduce_scatter", 2): "2026-08-23-2-abvalidate/reduce_scatter_2n_grid.validated.csv",
    # 2026-08-24 batch on {3,5,8}, job 20718
    ("all_reduce", 3): "2026-08-24-1-abvalidate3n/all_reduce_3n_grid.validated.csv",
    ("all_gather", 3): "2026-08-24-1-abvalidate3n/all_gather_3n_grid.validated.csv",
    ("reduce_scatter", 3): "2026-08-24-1-abvalidate3n/reduce_scatter_3n_grid.validated.csv",
}
NODESETS = {
    ("all_reduce", 1): "node 7 (2026-08-04)", ("all_reduce", 2): "{5,7} (08-04)",
    ("all_reduce", 3): "{5,6,7} (08-04)",
    ("broadcast", 1): "node 6 (08-18); defaults node 8 (08-17) - cross-node",
    ("broadcast", 2): "{1,3} (08-18)", ("broadcast", 3): "{1,3,6} (08-18)",
    ("reduce", 1): "node 6 (08-18); defaults node 8 (08-17) - cross-node",
    ("reduce", 2): "{1,3} (08-18)", ("reduce", 3): "{1,3,6} (08-18)",
    ("all_gather", 1): "node 7 (08-19)", ("all_gather", 2): "{5,7} (08-19)",
    ("all_gather", 3): "{5,6,7} (08-19)",
    ("reduce_scatter", 1): "node 7 (08-19)", ("reduce_scatter", 2): "{5,7} (08-19)",
    ("reduce_scatter", 3): "{5,6,7} (08-19)",
}


def size_h(b):
    for u, d in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if b >= d:
            v = b / d
            return f"{v:.0f}{u}" if v == int(v) else f"{v:.1f}{u}"
    return str(b)


def default_curve(coll, nodes):
    out = {}
    if coll == "all_reduce":
        with open(RT / f"2026-08-04-{ {1:1,2:4,3:6}[nodes] }-sweep{nodes}node/merged.csv".replace(" ", "")) as f:
            for row in csv.DictReader(f):
                if (row.get("requested_algo") or "").strip():
                    continue
                out.setdefault(int(row["size_bytes"]), []).append(float(row["busbw_ip"]))
        return {s: statistics.median(v) for s, v in out.items()}
    srcs = {"broadcast": ["2026-08-18-1-newcolls/newcolls_default_curves.csv",
                          "2026-08-17-4-hotspot-mn/hotspot_mn_curves.csv"],
            "reduce": ["2026-08-18-1-newcolls/newcolls_default_curves.csv",
                       "2026-08-17-4-hotspot-mn/hotspot_mn_curves.csv"],
            "all_gather": ["2026-08-19-2-agrs/agrs_default_curves.csv"],
            "reduce_scatter": ["2026-08-19-2-agrs/agrs_default_curves.csv"]}[coll]
    for src in srcs:
        with open(RT / src) as f:
            for row in csv.DictReader(f):
                if row["collective"] != f"{coll}_perf" or int(row["num_nodes"]) != nodes:
                    continue
                out.setdefault(int(row["size_bytes"]), []).append(float(row["busbw_ip"]))
        if out:
            break
    return {s: statistics.median(v) for s, v in out.items()}


def default_combo(coll, nodes):
    """size -> 'ALGO/PROTO (ch~N)' as RCCL chose by default. algo/proto are
    -A-verified; the channel count is the -A planned ceiling (finding 01), so
    it is shown with a tilde. 'mixed' when repeats disagreed on algo/proto."""
    rows = []
    if coll == "all_reduce":
        with open(RT / f"2026-08-04-{ {1:1,2:4,3:6}[nodes] }-sweep{nodes}node/merged.csv".replace(" ", "")) as f:
            for row in csv.DictReader(f):
                if (row.get("requested_algo") or "").strip():
                    continue
                rows.append((int(row["size_bytes"]), row["algo"], row["proto"],
                             row["nchannels"]))
    else:
        if coll in ("broadcast", "reduce"):
            if nodes == 1:  # hotspot-mn remote metrics, node 8 (cross-node)
                tag = {"broadcast": "b1", "reduce": "r1"}[coll]
                srcs = sorted((RT / "2026-08-17-4-hotspot-mn/remote").glob(
                    f"{tag}_r*/run_*/metrics.csv"))
            else:
                srcs = [RT / f"2026-08-18-1-newcolls/{coll}_{nodes}n_def_raw.txt"]
        else:
            srcs = [RT / f"2026-08-19-2-agrs/{coll}_{nodes}n_def_raw.txt"]
        import io
        for src in srcs:
            text = src.read_text()
            for chunk in text.split("== ")[0:] if src.suffix == ".txt" else [text]:
                body = chunk[chunk.find("collective,"):] if "collective," in chunk else ""
                if not body:
                    continue
                for row in csv.DictReader(io.StringIO(body)):
                    try:
                        if row["collective"] != f"{coll}_perf" or                            int(row["num_nodes"]) != nodes or                            (row.get("requested_algo") or "").strip():
                            continue
                        rows.append((int(row["size_bytes"]), row["algo"],
                                     row["proto"], row["nchannels"]))
                    except (KeyError, ValueError, TypeError):
                        continue
    out = {}
    for size in {r[0] for r in rows}:
        combos = [(a, p) for s2, a, p, c in rows if s2 == size]
        chs = [c for s2, a, p, c in rows if s2 == size]
        top = max(set(combos), key=combos.count)
        label = f"{top[0]}/{top[1]}"
        if len(set(combos)) > 1:
            label += " (mixed)"
        ch = max(set(chs), key=chs.count)
        out[size] = f"{label} ch~{ch}"
    return out


def live_spread(coll, nodes):
    """size -> 'def lo-hi / cfg lo-hi' from per_size_stats.csv (new A/B runs
    only; older validated dirs predate the sidecar)."""
    path = AB.get((coll, nodes))
    if not path:
        return {}
    import glob as _g
    base = (RT / path).parent
    out = {}
    for f in _g.glob(str(base / "**" / "per_size_stats.csv"), recursive=True):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if int(row["nodes"]) != nodes:
                    continue
                out[int(row["size_bytes"])] = {
                    "spread": (f"{row['def_min']}-{row['def_max']} / "
                               f"{row['cfg_min']}-{row['cfg_max']}"),
                    "def_exec": row.get("def_exec", ""),
                }
    return out


def winners(coll, nodes):
    out = {}
    with open(CMP / f"{coll}_{nodes}n_optimized.csv") as f:
        for row in csv.DictReader(f):
            out[int(row["size_bytes"])] = {
                "cfg": f"{row['algo']}/{row['proto']}/ch{row['nchannels']}",
                "bw": float(row["busbw_ip"]) if row["busbw_ip"] else None}
    return out


def live_verdicts(coll, nodes):
    """size -> (verdict, text) from the 08-19 validated configs."""
    path = AB.get((coll, nodes))
    if not path:
        return {}
    kept, dropped = [], []
    with open(RT / path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"# dropped: \w+,(\d+),(\d+),.*?\((.*)\)$", line)
            if m:
                dropped.append((int(m.group(1)), int(m.group(2)), m.group(3)))
            elif line and not line.startswith("#") and not line.startswith("collective_type"):
                p = line.split(",")
                kept.append((int(p[1]), int(p[2])))
    return {"kept": kept, "dropped": dropped}


CSS = """body{font-family:-apple-system,system-ui,sans-serif;margin:12px;max-width:960px;
background:#14161a;color:#d8dbe0}
table{border-collapse:collapse;font-size:13px;width:100%;display:block;overflow-x:auto}
th,td{border:1px solid #34383f;padding:4px 6px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
tr:nth-child(even){background:#1b1e24}
h1{font-size:20px;color:#f0f2f5}h2{font-size:16px;color:#e6e9ee}h3{font-size:14px;color:#e6e9ee}
.g{color:#4ec96a;font-weight:600}.b{color:#ff6b6b}.n{color:#8a8f98}
.v{background:#173321}.x{background:#331a1a}
.note{color:#9aa0a8;font-size:13px}.tag{font-size:11px;color:#8a8f98}
a{color:#6ea8ff;text-decoration:none}
pre{background:#1b1e24;padding:8px;font-size:12px;overflow-x:auto;border-radius:6px;color:#d8dbe0}"""


def page_head(title):
    return (f"<!doctype html><meta name=viewport content='width=device-width,"
            f"initial-scale=1'><title>{title}</title><style>{CSS}</style>"
            f"<p><a href='index.html'>&larr; index</a></p><h1>{title}</h1>")


def build_gains(coll):
    h = [page_head(f"{coll}: ours vs RCCL default")]
    h.append("<p class=note>gain = (our config &minus; default) / default, on busbw. "
             "'default combo' = what RCCL itself chose (algo/proto verified via -A 1; "
             "ch~ is the planned ceiling, not the verified actual - finding 01). "
             "<b>predicted</b> = grid data (env-var path). <b>validated</b> = live A/B "
             "through the tuner plugin. A predicted win is not deliverable until validated: "
             "the env-var path and the plugin path are proven non-equivalent.</p>")
    for nodes in (1, 2, 3):
        d, w = default_curve(coll, nodes), winners(coll, nodes)
        dc = default_combo(coll, nodes)
        sp = live_spread(coll, nodes)
        lv = live_verdicts(coll, nodes)
        h.append(f"<h2>{nodes} node{'s' if nodes > 1 else ''} "
                 f"<span class=tag>({NODESETS[(coll, nodes)]})</span></h2>")
        h.append("<table><tr><th>size</th><th>default combo</th><th>default</th>"
                 "<th>our config</th><th>ours busbw</th><th>gain</th>"
                 "<th>A/B spread (def / cfg)</th><th>status</th></tr>")
        for s in sorted(d):
            dv = d[s]
            if s in w and w[s]["bw"] is not None:
                bw, cfg = w[s]["bw"], w[s]["cfg"]
                gain = (bw - dv) / dv * 100.0
                cls = "g" if gain > 2 else ("b" if gain < -2 else "n")
                status, rowcls = "predicted", ""
                if lv:
                    for lo, hi in lv["kept"]:
                        if lo <= s <= hi:
                            status, rowcls = "VALIDATED win (live)", " class=v"
                    for lo, hi, why in lv["dropped"]:
                        if lo <= s <= hi:
                            status, rowcls = f"live A/B dropped: {why}", " class=x"
                h.append(f"<tr{rowcls}><td>{size_h(s)}</td>"
                         f"<td>{(sp.get(s) or {}).get('def_exec') or dc.get(s, '?')}</td>"
                         f"<td>{dv:g}</td><td>{cfg}</td>"
                         f"<td>{bw:g}</td><td class={cls}>{gain:+.1f}%</td>"
                         f"<td class=n>{(sp.get(s) or {}).get('spread', '')}</td>"
                         f"<td>{html.escape(status)}</td></tr>")
            else:
                h.append(f"<tr><td>{size_h(s)}</td><td>{dc.get(s, '?')}</td><td>{dv:g}</td>"
                         f"<td colspan=4 class=n>not tunable (RCCL Direct kernel below "
                         f"threshold)</td><td class=n>default rules</td></tr>")
        h.append("</table>")
    if coll == "all_gather":
        h.append("<p class=note>all_gather below 16M (1n) / 8M (2-3n) always runs RCCL's "
                 "AMD-specific Direct kernel regardless of any request - those sizes have "
                 "no rules and keep the default.</p>")
    (HERE / f"gains_{coll}.html").write_text("\n".join(h))


def build_index():
    h = [f"<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>GPU-107 tuning</title><style>{CSS}</style>"]
    h.append("<h1>GPU-107 RCCL tuning — status &amp; results</h1>")
    h.append("<h2>The flow</h2><pre>book nodes (human)\n"
             "  -> adaptive search, all sizes      adaptive_search.py live\n"
             "  -> config emitted                  --emit-optimized -> generate_tuner_config.py\n"
             "  -> A/B validate vs default         validate_tuner_config.py (plugin path, psup)\n"
             "  -> gains pages                     this site</pre>")
    h.append("<p class=note>grid = calibration only (truth-maker, rare). "
             "optuna/random = evaluation baselines, not the flow. triage = parked "
             "(possible monitor later).</p>")
    h.append("<h2>Gains vs RCCL default (per collective)</h2><ul>")
    for c in COLLS:
        h.append(f"<li><a href='gains_{c}.html'>{c}</a></li>")
    h.append("<li>alltoall — no config: RCCL forces RING/SIMPLE at source; "
             "channel sweep pending (live batch)</li></ul>")
    h.append("<h2>Method evaluation</h2><ul>"
             "<li><a href='methods.html'>grid vs adaptive vs optuna vs random vs triage"
             "</a> — 15 grids, offline replay</li>"
             "<li><a href='configs.html'>the 15 per-scale configs + validation status</a></li>"
             "<li><a href='drafts/new-facts.md'>10 new verified facts (draft)</a></li>"
             "<li><a href='explanations/adaptive-walkthrough.md'>how adaptive works</a> · "
             "<a href='explanations/optuna-walkthrough.md'>optuna</a> · "
             "<a href='explanations/triage-walkthrough.md'>triage</a> · "
             "<a href='explanations/optuna-verdict.md'>optuna verdict</a></li></ul>")
    h.append("<h2>Validation state</h2><p class=note>Live A/B verdicts on 11 of 15 "
             "scales: 2026-08-19 (broadcast 3n, reduce 2n/3n on {5,7}/{5,6,7}) and "
             "2026-08-23 (all_reduce 1n/2n, broadcast 1n, reduce 1n, all_gather 1n/2n, "
             "reduce_scatter 1n/2n on {5,8}, job 20713, via ab_run.py). all_gather 1n and "
             "reduce_scatter 1n: NO rule survived - RCCL defaults win, no config ships "
             "(all_gather 1n's only rule hid a -59.7% landmine). broadcast 2n: "
             "unmeasurable - 18 preflight-failed attempts across two node pairs "
             "({5,7} and {5,8}); intrinsic small-size noise, needs a gate decision. "
             "2026-08-24: all_reduce/all_gather/reduce_scatter 3n validated on {3,5,8} "
             "(job 20718) - every scale except broadcast 2n now has a live verdict.</p>")
    (HERE / "index.html").write_text("\n".join(h))


for coll in COLLS:
    build_gains(coll)
build_index()
print("pages built")
