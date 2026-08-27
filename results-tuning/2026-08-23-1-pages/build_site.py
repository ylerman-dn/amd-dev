#!/usr/bin/env python3
"""The single results view: one page per collective, freshest verdicts only.

One data resolver scans every rccl-tune results dir (pilot/stage*) and picks,
per (collective, scale), the newest run that produced an A/B verdict. No
hand-maintained map - the scan IS the source of truth. Each collective page
carries: status strip, and per scale the provenance line, the search config
with colored verdicts, the final shipped config, the per-size table with
executed-truth values and live A/B numbers, and the raw runs (collapsed).
"""
import csv, glob, html, json, re
from pathlib import Path

HERE = Path(__file__).parent
RT = HERE.parent
COLLS = ["all_reduce", "broadcast", "reduce", "all_gather", "reduce_scatter",
         "alltoall"]
SCALES = [1, 2, 3, 4, 5]
KNOWN_ISSUES = {
    ("broadcast", 2): "No live verdict is possible under the current noise "
        "gates: 22 A/B attempts across three node pairs ({5,7}, {5,8}, {1,3}) "
        "failed preflight on intrinsic small-size (4-16K) scatter. RESOLVED "
        "2026-08-26 by the rule-scoped preflight: 4K/16K excluded as "
        "unmeasurable, every other size judged - 4 rules verified.",
    ("broadcast", 4): "still unverdicted after 10 voided attempts across 3 days "
        "(rc=2/rc=3), including on the same-leaf L1 set - transient noise, "
        "source unknown.",
    ("all_gather", 4): "A/B window-limited; one rc=2 void attempt quarantined.",
    ("reduce_scatter", 4): "A/B window-limited (4 refusals, same window).",
    ("alltoall", 1): "candidate ch=40 A/B'd 2026-08-26: no rule survived - "
        "defaults win. 32K excluded by the scoped preflight (chronic 690% "
        "same-config scatter).",
    ("alltoall", 4): "A/B voided (rc=2) in all 4 attempts on 2026-08-26 - "
        "transient fabric stalls at 4+ nodes (see stage5n NOISE-ANALYSIS.md). "
        "Numbers below are from the voided runs; no verdicts were issued.",
}
CSS = """body{font-family:-apple-system,system-ui,sans-serif;margin:12px;max-width:1280px;
background:#14161a;color:#d8dbe0}
.tw{overflow-x:auto;margin:10px 0}
table{border-collapse:collapse;font-size:13px;width:100%}
th{border:1px solid #34383f;padding:4px 8px;text-align:center}
td{border:1px solid #34383f;padding:4px 8px;text-align:right;white-space:nowrap}
td:first-child{text-align:left}
table.fit{width:auto}
table.fit td{text-align:left;white-space:normal}
td.runs{font-family:ui-monospace,monospace;font-size:11.5px;max-width:430px}
td.wrap{white-space:normal;min-width:160px;max-width:260px;text-align:left;font-size:12px}
h1{font-size:20px;color:#f0f2f5}h2{font-size:16px;color:#e6e9ee;margin-top:26px}
h3{font-size:14px;color:#e6e9ee}
.g{color:#4ec96a;font-weight:600}.b{color:#ff6b6b}.n{color:#8a8f98}
.v{background:#173321}.x{background:#331a1a}.warn{color:#e0c05a}
.note{color:#9aa0a8;font-size:13px}
a{color:#6ea8ff;text-decoration:none}
pre{background:#1b1e24;padding:8px;font-size:12px;overflow-x:auto;border-radius:6px;line-height:1.5}
pre .keep{color:#4ec96a}pre .drop{color:#ff6b6b;text-decoration:line-through}
pre .why{color:#8a8f98;text-decoration:none;font-style:italic}
details{margin:8px 0}summary{cursor:pointer;color:#6ea8ff;font-size:13px}"""


def size_h(b):
    for u, d in (("M", 1 << 20), ("K", 1 << 10)):
        if b >= d:
            v = b / d
            return f"{v:.0f}{u}" if v == int(v) else f"{v:.1f}{u}"
    return str(b)


def short_why(why):
    m = re.search(r"regression of (-[\d.]+)%", why)
    if m:
        return f"regression {m.group(1)}%"
    m = re.search(r"best gain only ([+-][\d.]+)%", why)
    if m:
        return f"no gain ({m.group(1)}%)"
    m = re.search(r"P\(sup\) only ([\d.]+)", why)
    if m:
        return f"not reproducible (P {m.group(1)})"
    return why[:40]


def stem_of(coll, scale):
    return f"{coll}_{scale}n".replace("all_reduce", "allreduce").replace(
        "all_gather", "allgather").replace("reduce_scatter", "reducescatter")


def resolve():
    """(coll, scale) -> dict with the freshest run's artifact paths."""
    out = {}
    for d in sorted(RT.glob("2026-*-*")):
        if not (d.is_dir() and re.search(r"-(pilot|stage\w*|a2a\w*)$", d.name)):
            continue
        for coll in COLLS:
            for scale in SCALES:
                tag = f"{coll}_{scale}n"
                rep = d / tag / "adaptive_report.json"
                if not rep.exists():
                    continue
                vfile = None
                for cand in (d / f"{tag}_persize.validated.csv",
                             d / f"{tag}.validated.csv"):
                    if cand.exists():
                        vfile = cand
                conf = None
                for cand in (d / f"{tag}_persize.conf", d / f"{tag}.conf"):
                    if cand.exists():
                        conf = cand
                        break
                stats = sorted(d.glob(f"ab_out*/{stem_of(coll, scale)}*/"
                                      "per_size_stats.csv"))
                prev = out.get((coll, scale))
                # newest dir wins; a run WITH a verdict beats one without
                if prev and prev["vfile"] and not vfile:
                    continue
                out[(coll, scale)] = dict(dir=d, report=rep, vfile=vfile,
                                          conf=conf,
                                          stats=stats[-1] if stats else None)
    return out


def rules(vfile):
    kept, dropped = [], []
    for line in open(vfile):
        line = line.strip()
        m = re.match(r"# dropped: (\w+,(\d+),(\d+),.*?)\s+\((.*)\)$", line)
        if m:
            dropped.append((m.group(1), int(m.group(2)), int(m.group(3)),
                            m.group(4)))
        elif line and not line.startswith("#") and \
                not line.startswith("collective_type"):
            p = line.split(",")
            kept.append((line, int(p[1]), int(p[2])))
    return kept, dropped


def load_stats(f, nodes):
    out = {}
    with open(f) as fh:
        for row in csv.DictReader(fh):
            if int(row["nodes"]) == nodes:
                out[int(row["size_bytes"])] = row
    return out


def conf_block(conf, kept, dropped):
    kept_raw = {k[0] for k in kept}
    lines = []
    for line in open(conf):
        line = line.rstrip()
        if line.startswith("collective_type"):
            lines.append(f"<span class=n>{html.escape(line)}</span>")
            continue
        hit = False
        for raw, lo, hi, why in dropped:
            if line == raw:
                lines.append(f"<span class=drop>{html.escape(line)}</span>"
                             f"  <span class=why>&larr; {html.escape(short_why(why))}</span>")
                hit = True
        if not hit:
            cls = "keep" if line in kept_raw else "n"
            mark = "  <span class=why>&larr; kept</span>" if line in kept_raw else ""
            lines.append(f"<span class={cls}>{html.escape(line)}</span>{mark}")
    return "<pre>" + "\n".join(lines) + "</pre>"


def scale_status(r, coll, scale):
    if not r:
        return ("not run", "n")
    if not r["vfile"]:
        sv = r["dir"] / f"{coll}_{scale}n.SEARCH-VERDICT.txt"
        if sv.exists():
            return ("defaults win (search)", "n")
        return ("no live verdict", "warn")
    kept, _ = rules(r["vfile"])
    if not kept:
        return ("defaults win", "n")
    return (f"{len(kept)} rules validated", "g")


R = resolve()
status = {}
for coll in COLLS:
    page = [f"<!doctype html><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'>"
            f"<title>{coll}</title><style>{CSS}</style>"]
    page.append(f"<p><a href='index.html'>&larr; all collectives</a></p>")
    page.append(f"<h1>{coll}</h1>")
    if coll == "alltoall":
        page.append(
            "<p class=note>alltoall never reaches RCCL's collective tuner: "
            "it is decomposed into p2p send/recv tasks at enqueue, so "
            "NCCL_ALGO/NCCL_PROTO requests are no-ops, <code>-A 1</code> "
            "prints N/A, and the tuner plugin cannot apply per-size rules. "
            "The search therefore sweeps the channel request only "
            "('P2P/-/N'), the deployable artifact is one global "
            "NCCL_MIN/MAX_NCHANNELS value, and the A/B validates that env "
            "setting against the untouched default. Executed channels are "
            "read from the debug logs' 'p2p channels' line.</p>")
    strip = []
    for scale in SCALES:
        txt, cls = scale_status(R.get((coll, scale)), coll, scale)
        status[(coll, scale)] = (txt, cls)
        strip.append(f"{scale}n: <span class={cls}>{txt}</span>")
    page.append("<p class=note>" + " &nbsp;·&nbsp; ".join(strip) + "</p>")
    for scale in SCALES:
        r = R.get((coll, scale))
        if not r:
            continue
        rep = json.load(open(r["report"]))
        page.append(f"<h2>{scale} node{'s' if scale > 1 else ''}</h2>")
        cost = ""
        if "full_runs" in rep:
            cost = (f"search {rep['runs']} runs vs grid {rep['full_runs']} "
                    f"(&minus;{rep['saving_pct']:.0f}%), ")
        page.append(f"<p class=note>run <b>{r['dir'].name}</b> · "
                    f"{cost}{rep.get('configs', '?')} configs, "
                    f"{rep.get('search_wall_s', 0) / 60:.0f} min search · "
                    f"9 A/B repeats/arm</p>")
        if (coll, scale) in KNOWN_ISSUES:
            page.append(f"<p class=warn>&#9888; "
                        f"{html.escape(KNOWN_ISSUES[(coll, scale)])}</p>")
        kept, dropped = rules(r["vfile"]) if r["vfile"] else ([], [])
        if r["conf"] and r["vfile"]:
            page.append("<h3>Search config &rarr; A/B verdicts</h3>")
            page.append(conf_block(r["conf"], kept, dropped))
            final = r["dir"] / f"{coll}_{scale}n.final.conf"
            if final.exists() and kept:
                page.append("<h3>Final shipped config</h3>"
                            f"<pre class=keep>{html.escape(final.read_text().strip())}</pre>")
            elif r["vfile"] and not kept:
                page.append("<p class=note>No config ships at this scale - "
                            "RCCL's defaults won every size.</p>")
        st = load_stats(r["stats"], scale) if r["stats"] else {}
        if not st and rep.get("winners"):
            # no live verdict: same headers as validated tables; A/B-only
            # columns stay blank. Executed combos parsed from the SEARCH
            # session's own debug logs (exec_search.csv).
            defaults = {}
            for d_run in rep.get("default_busbw", []):
                for k, v in d_run.items():
                    defaults.setdefault(int(k), []).append(v)
            execs = {}
            ef = r["dir"] / f"{coll}_{scale}n" / "exec_search.csv"
            if ef.exists():
                with open(ef) as fh:
                    for row in csv.DictReader(fh):
                        execs[int(row["size_bytes"])] = row
            page.append("<h3>Per size (predicted from the search - "
                        "<span class=b>NOT validated, does not ship</span>)</h3>")
            page.append("<p class=note>Search measurements (env-var path); "
                        "executed combos from the search runs' debug logs. "
                        "P(sup)/spread/verdict require an A/B and stay blank.</p>")
            page.append("<div class=tw><table><tr><th>size</th>"
                        "<th>default executed</th><th>ours requested</th>"
                        "<th>ours executed</th><th>default med</th>"
                        "<th>ours med</th><th>gain</th><th>P(sup)</th>"
                        "<th>spread def / cfg</th><th>verdict</th></tr>")
            import statistics as _st
            for s2, w in sorted(((int(k), v) for k, v in
                                 rep["winners"].items())):
                dmed = (_st.median(defaults[s2])
                        if defaults.get(s2) else None)
                bw = w.get("busbw")
                e = execs.get(s2, {})
                if dmed and bw:
                    g = (bw / dmed - 1) * 100
                    gcls = "g" if g > 2 else ("b" if g < -2 else "n")
                    gtxt = f"<td class={gcls}>{g:+.1f}%</td>"
                    dtxt = f"<td>{dmed:g}</td><td>{bw:g}</td>"
                else:
                    gtxt, dtxt = "<td class=n></td>", "<td class=n></td><td class=n></td>"
                page.append(f"<tr><td>{size_h(s2)}</td>"
                            f"<td>{e.get('def_exec', '')}</td>"
                            f"<td>{w['cfg']}</td>"
                            f"<td>{e.get('ours_exec', '')}</td>"
                            f"{dtxt}{gtxt}"
                            f"<td class=n></td><td class=n></td>"
                            f"<td class='wrap n'>no A/B</td></tr>")
            page.append("</table></div>")
        if st:
            if r["vfile"]:
                page.append("<h3>Per size (live A/B numbers)</h3>")
            else:
                page.append("<h3>Per size (live A/B numbers - run VOIDED, "
                            "<span class=b>no verdicts issued</span>)</h3>")
            page.append("<div class=tw><table><tr><th>size</th><th>default executed</th>"
                        "<th>ours requested</th><th>ours executed</th>"
                        "<th>default med</th><th>ours med</th><th>gain</th>"
                        "<th>P(sup)</th><th>spread def / cfg</th>"
                        "<th>verdict</th></tr>")
            winners = {int(s): w["cfg"] for s, w in rep["winners"].items()}
            for s in sorted(winners):
                row = st.get(s)
                if not row:
                    continue
                verdict, cls = "no rule", " "
                for raw, lo, hi in kept:
                    if lo <= s <= hi:
                        verdict, cls = "KEPT", " class=v"
                for raw, lo, hi, why in dropped:
                    if lo <= s <= hi:
                        verdict, cls = f"dropped: {short_why(why)}", " class=x"
                gain = float(row["gain_pct"]) if row["gain_pct"] else 0.0
                gcls = "g" if gain > 2 else ("b" if gain < -2 else "n")
                cexec = row.get("cfg_exec", "")
                if row.get("ch_trimmed") == "yes":
                    cexec += " (trimmed)"
                page.append(
                    f"<tr{cls}><td>{size_h(s)}</td>"
                    f"<td>{row.get('def_exec', '')}</td>"
                    f"<td>{winners[s]}</td><td>{cexec}</td>"
                    f"<td>{row['def_median']}</td><td>{row['cfg_median']}</td>"
                    f"<td class={gcls}>{gain:+.1f}%</td>"
                    f"<td>{float(row['psup']):.2f}</td>"
                    f"<td class=n>{row['def_min']}-{row['def_max']} / "
                    f"{row['cfg_min']}-{row['cfg_max']}</td>"
                    f"<td class=wrap>{html.escape(verdict)}</td></tr>")
            page.append("</table></div>")
            page.append("<details><summary>raw A/B runs</summary>"
                        "<div class=tw><table class=fit><tr><th>size</th><th>arm</th><th>runs</th>"
                        "<th>median</th></tr>")
            for s in sorted(st):
                row = st[s]
                for arm, key, med in (("default", "def_runs", row["def_median"]),
                                      ("config", "cfg_runs", row["cfg_median"])):
                    page.append(f"<tr><td>{size_h(s)}</td><td>{arm}</td>"
                                f"<td class=runs>{row[key].replace(';', ' ')}</td>"
                                f"<td><b>{med}</b></td></tr>")
            page.append("</table></div></details>")
    (HERE / f"collective_{coll}.html").write_text("\n".join(page))

# index: status matrix + satellites
idx = [f"<!doctype html><meta charset=utf-8><meta name=viewport "
       f"content='width=device-width,initial-scale=1'>"
       f"<title>GPU-107 tuning</title><style>{CSS}</style>"]
idx.append("<h1>RCCL tuning — results by collective</h1>")
idx.append("<p class=note>Freshest live A/B verdicts, resolved automatically "
           "from all rccl-tune runs. 1 measurement = median of 3 runs; A/B = "
           "9 repeats/arm through the tuner plugin.</p>")
idx.append("<div class=tw><table><tr><th>collective</th>"
           + "".join(f"<th>{s}n</th>" for s in SCALES) + "</tr>")
for coll in COLLS:
    cells = "".join(
        f"<td class={status[(coll, s)][1]}>{status[(coll, s)][0]}</td>"
        for s in SCALES)
    idx.append(f"<tr><td><a href='collective_{coll}.html'>{coll}</a></td>"
               f"{cells}</tr>")
idx.append("</table></div>")
idx.append("<p class=note>alltoall: runs on the p2p path (send/recv tasks) - "
           "NCCL_ALGO/NCCL_PROTO are no-ops and the tuner plugin is never "
           "consulted, so its search is channels-only and its A/B uses "
           "NCCL_MIN/MAX_NCHANNELS as the ON arm (in the CLI since "
           "2026-08-26).</p>")
idx.append("<h2>Method evaluation &amp; docs</h2><ul>"
           "<li><a href='approaches.html'>the story: grid vs adaptive vs "
           "optuna vs random vs triage</a> "
           "(<a href='approaches-draft.html'>style-selection draft</a>)</li>"
           "<li><a href='methods.html'>full offline comparison, 15 grids</a></li>"
           "<li><a href='explanations/adaptive-walkthrough.md'>adaptive "
           "walkthrough</a> · <a href='explanations/optuna-walkthrough.md'>"
           "optuna</a> · <a href='explanations/triage-walkthrough.md'>triage"
           "</a> · <a href='explanations/optuna-verdict.md'>optuna verdict</a></li>"
           "<li><a href='drafts/new-facts.md'>10 verified facts (awaiting "
           "approval)</a></li>"
           "<li><a href='handoff.md'>handoff / summary</a> · "
           "<a href='cli/corrections.md'>corrections log</a></li></ul>")
(HERE / "index.html").write_text("\n".join(idx))
print("site built:", [f"collective_{c}.html" for c in COLLS])
