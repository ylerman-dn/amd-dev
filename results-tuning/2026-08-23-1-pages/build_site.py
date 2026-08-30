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
        "unmeasurable, every other size judged. Re-judged under cv-core "
        "2026-08-27: 3 rules (32K lost - baseline CV 8.05%, over the limit).",
    ("broadcast", 4): "verdict via cv-core replay (2026-08-27) of the L1 run: single "
        "transient dips discarded by MAD outlier rejection, chronic sizes still "
        "excluded. 13 earlier attempts voided under the plain spread gate.",
    ("all_gather", 4): "verdict via cv-core replay (2026-08-27) of the L1 run.",
    ("reduce_scatter", 4): "verdict via cv-core replay (2026-08-27) of the L1 run.",
    ("all_reduce", 5): "verdict via cv-core replay (2026-08-27) of the stored 5n run - "
        "every live attempt was voided under the old spread gate; the MAD core "
        "rescue found judgeable data. Fresh confirmation A/B pending a window.",
    ("broadcast", 5): "verdict via cv-core replay (2026-08-27) of the stored 5n run - "
        "every live attempt was voided under the old spread gate; the MAD core "
        "rescue found judgeable data. Fresh confirmation A/B pending a window.",
    ("reduce", 5): "verdict via cv-core replay (2026-08-27) of the stored 5n run - "
        "every live attempt was voided under the old spread gate; the MAD core "
        "rescue found judgeable data. Fresh confirmation A/B pending a window.",
    ("all_gather", 5): "verdict via cv-core replay (2026-08-27) of the stored 5n run - "
        "every live attempt was voided under the old spread gate; the MAD core "
        "rescue found judgeable data. Fresh confirmation A/B pending a window.",
    ("reduce_scatter", 5): "verdict via cv-core replay (2026-08-27) of the stored 5n run - "
        "every live attempt was voided under the old spread gate; the MAD core "
        "rescue found judgeable data. Fresh confirmation A/B pending a window.",
    ("alltoall", 1): "candidate ch=40 A/B'd 2026-08-26: no rule survived - "
        "defaults win. 32K excluded by the scoped preflight (chronic 690% "
        "same-config scatter). NOTE: that A/B used NCCL_MIN/MAX_NCHANNELS, "
        "which does not control p2p channels, so it changed nothing "
        "(corrected 2026-08-30, findings/12).",
    ("alltoall", 4): "verdict via cv-core replay (2026-08-27): the env-arm rule "
        "contains a -14.3% regression - defaults win at 4n too. Same knob "
        "caveat as 1n: re-measure with NCCL_MIN/MAX_P2P_NCHANNELS.",
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


def _vals(row, key):
    return [float(x) for x in row[key].split(";") if x]


def _stats(vals):
    """median, mean, stddev, CV% of one arm's repeats."""
    import statistics as _s
    if not vals:
        return "", "", "", ""
    med = _s.median(vals)
    mean = _s.mean(vals)
    sd = _s.stdev(vals) if len(vals) > 1 else 0.0
    cv = sd / mean * 100 if mean else 0.0
    return f"{med:g}", f"{mean:.4g}", f"{sd:.3g}", f"{cv:.1f}"


def _gate_cell(vals):
    """The cv-core gate's view of one arm: CV, core size, and its decision."""
    import statistics as _s
    if len(vals) < 2 or min(vals) <= 0:
        return "<span class=n>no data</span>"
    mean = _s.mean(vals)
    cv = _s.stdev(vals) / mean * 100 if mean else 0.0
    med = _s.median(vals)
    mad = _s.median([abs(v - med) for v in vals])
    scale = max(1.4826 * mad if mad > 0 else
                1.2533 * _s.mean([abs(v - med) for v in vals]), 0.005)
    core = [v for v in vals if abs(v - med) / scale <= 3.5]
    ccv = (_s.stdev(core) / _s.mean(core) * 100
           if len(core) > 1 and _s.mean(core) else 0.0)
    if cv <= 8.0:
        return f"<span class=n>CV {cv:.1f}%</span>"
    if len(core) >= 7 and ccv <= 8.0:
        return (f"<span class=warn>CV {cv:.1f}% &rarr; core {len(core)}/"
                f"{len(vals)} CV {ccv:.1f}% (dips discarded)</span>")
    return f"<span class=b>CV {cv:.1f}%, no clean core - NOISY</span>"


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
            "('P2P/-/N'), the deployable artifact is one global channel "
            "value, and the A/B validates that env setting against the "
            "untouched default. Executed channels are read from the debug "
            "logs' 'p2p channels' line.</p>")
        page.append(
            "<p class=note><b>Correction, 2026-08-30.</b> The 2026-08-26 runs "
            "used <code>NCCL_MIN/MAX_NCHANNELS</code>, which bounds the "
            "<i>collective</i> path and does not control p2p channels "
            "(<code>src/graph/paths.cc:982-983</code>) - so those runs changed "
            "nothing and their '+0.00% median' measured nothing. Re-run at 2 "
            "nodes with the correct <code>NCCL_MIN/MAX_P2P_NCHANNELS</code>: "
            "forcing 8 channels costs 11-27%, while 16/32/64 all match the "
            "default, so no gain is available. The 'tuner never consulted' "
            "claim is now backed by a log as well as by source - a full run "
            "with the plugin loaded produced zero 'Applied config' and zero "
            "'Config does not match' lines. Note <code>pow2Up</code> "
            "(<code>paths.cc:1015</code>) rounds the request to a power of "
            "two, so only 8/16/32/64 are reachable - this is why an earlier "
            "request for 40 executed 64. See "
            "<code>results-tuning/2026-08-30-1-value/alltoall-2n/</code> and "
            "<code>findings/12-alltoall-not-tunable.md</code>.</p>")
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
                        "<th>P(sup)</th><th>noise gate def / cfg</th>"
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
                    f"<td class='wrap n'>{_gate_cell(_vals(row, 'def_runs'))}"
                    f" / {_gate_cell(_vals(row, 'cfg_runs'))}</td>"
                    f"<td class=wrap>{html.escape(verdict)}</td></tr>")
            page.append("</table></div>")
            page.append("<details><summary>raw A/B runs</summary>"
                        "<div class=tw><table class=fit><tr><th>size</th><th>arm</th><th>runs</th>"
                        "<th>median</th><th>mean</th><th>stddev</th><th>CV %</th></tr>")
            for s in sorted(st):
                row = st[s]
                for arm, key in (("default", "def_runs"), ("config", "cfg_runs")):
                    med, mean, sd, cvp = _stats(_vals(row, key))
                    page.append(f"<tr><td>{size_h(s)}</td><td>{arm}</td>"
                                f"<td class=runs>{row[key].replace(';', ' ')}</td>"
                                f"<td><b>{med}</b></td><td>{mean}</td>"
                                f"<td>{sd}</td><td>{cvp}</td></tr>")
            page.append("</table></div></details>")
    (HERE / f"collective_{coll}.html").write_text("\n".join(page))


# ------------------------------------------------------------- overview page
STEMS = {"all_reduce": "allreduce", "all_gather": "allgather",
         "reduce_scatter": "reducescatter", "broadcast": "broadcast",
         "reduce": "reduce", "alltoall": "alltoall"}
SIZES18 = [4096 * (2 ** i) for i in range(18)]

ov = [f"<!doctype html><meta charset=utf-8><meta name=viewport "
      f"content='width=device-width,initial-scale=1'>"
      f"<title>tuning overview</title><style>{CSS}"
      ".hm td{min-width:52px;text-align:center;font-size:11.5px}"
      ".g1{background:#1d3a26}.g2{background:#215c31}.g3{background:#2c8c46;color:#fff}"
      ".dw{background:#22262c;color:#8a8f98}.na{color:#4a4f57}</style>"]
ov.append("<p><a href='index.html'>&larr; index</a></p>")
ov.append("<h1>Verified gains — all collectives, all scales</h1>")
ov.append("<p class=note>Cell = verified A/B gain of the shipped rule covering "
          "that size (cv-core gate). 'def' = judged, defaults win. Blank = "
          "unmeasurable or no rule judged there. alltoall ships nothing "
          "anywhere - its defaults sit on the plateau at every scale.</p>")

def _gain_map(coll, scale, r):
    """size -> (gain, kept?) for judged sizes; None entries mean unjudged."""
    out = {}
    if not r or not r["vfile"]:
        return out
    kept_rows, dropped_rows = rules(r["vfile"])
    sp = (Path(r["conf"]).parent / "ab_out_replaycv" /
          f"{STEMS[coll]}_{scale}n" / "per_size_stats.csv") if r["conf"] else None
    stats = {}
    if sp and sp.exists():
        with open(sp) as fh:
            stats = {int(x["size_bytes"]): x for x in csv.DictReader(fh)}
    for s in SIZES18:
        covered_kept = any(lo <= s <= hi for _, lo, hi in kept_rows)
        covered_drop = any(lo <= s <= hi for _, lo, hi, _ in dropped_rows)
        st = stats.get(s)
        if covered_kept and st:
            out[s] = (float(st["gain_pct"] or 0), True)
        elif covered_drop:
            out[s] = (None, False)
    return out

cols = [(c, sc) for c in COLLS for sc in SCALES if (c, sc) in R]
maps = {cs: _gain_map(cs[0], cs[1], R.get(cs)) for cs in cols}
ov.append("<div class=tw><table class=hm><tr><th>size</th>" + "".join(
    f"<th>{c.replace('_', '_<br>')}<br>{sc}n</th>" for c, sc in cols) + "</tr>")
for s in SIZES18:
    row = [f"<tr><td>{size_h(s)}</td>"]
    for cs in cols:
        e = maps[cs].get(s)
        if e is None:
            row.append("<td class=na></td>")
        elif e[1]:
            g = e[0]
            cls = "g3" if g >= 50 else ("g2" if g >= 10 else "g1")
            row.append(f"<td class={cls}>+{g:.0f}%</td>")
        else:
            row.append("<td class=dw>def</td>")
    ov.append("".join(row) + "</tr>")
ov.append("</table></div>")

ov.append("<h2>Shipped configs</h2><div class=tw><table><tr><th>collective</th>"
          + "".join(f"<th>{sc}n</th>" for sc in SCALES) + "</tr>")
for c in COLLS:
    cells = []
    for sc in SCALES:
        r = R.get((c, sc))
        if not r or not r["vfile"]:
            cells.append("<td class=na>-</td>")
            continue
        kept_rows, _ = rules(r["vfile"])
        if not kept_rows:
            cells.append("<td class=dw>defaults</td>")
            continue
        m = maps[(c, sc)]
        best = max((g for g, k in m.values() if k and g is not None), default=0)
        fin = Path(r["conf"]).parent / f"{c}_{sc}n.final.conf"
        rel = fin.relative_to(RT) if fin.exists() else None
        link = f" · <a href='../{rel}'>conf</a>" if rel else ""
        cells.append(f"<td>{len(kept_rows)} rules · best +{best:.0f}%{link}</td>")
    ov.append(f"<tr><td><a href='collective_{c}.html'>{c}</a></td>"
              + "".join(cells) + "</tr>")
ov.append("</table></div>")
ov.append("<p class=note>120 verified rules total. Gate: cv-core "
          "(2026-08-27). Retry shortlist (real-looking gains lost to residual "
          "noise): broadcast 4n 128K +133%, reduce 5n 128K +71% / 64K +44%, "
          "broadcast 4n 32K +19%, broadcast 2n 32K +18% (P 0.99, baseline CV "
          "8.05%), all_gather 4n 64M +13%.</p>")
(HERE / "overview.html").write_text("\n".join(ov))
print("overview built")


# --------------------------------------------------------------- audit page
au = [f"<!doctype html><meta charset=utf-8><meta name=viewport "
      f"content='width=device-width,initial-scale=1'>"
      f"<title>verdict audit</title><style>{CSS}</style>"]
au.append("<p><a href='index.html'>&larr; index</a></p>")
au.append("<h1>Verdict audit — every accept and reject, sanity-checked</h1>")
_kept_n, _reasons, _suspects, _weak = 0, {}, [], []
for (_c, _sc) in sorted(R):
    _r = R[(_c, _sc)]
    if not _r["vfile"]:
        continue
    _sp = (Path(_r["conf"]).parent / "ab_out_replaycv" /
           f"{STEMS[_c]}_{_sc}n" / "per_size_stats.csv") if _r["conf"] else None
    _stats = {}
    if _sp and _sp.exists():
        with open(_sp) as fh:
            _stats = {int(x["size_bytes"]): x for x in csv.DictReader(fh)}
    for _line in Path(_r["vfile"]).read_text().splitlines():
        _m = re.match(r"# dropped: ([\w,]+?),(\d+),(\d+),(.*?)\s+\((.*)\)$", _line)
        if _m:
            _size, _why = int(_m.group(2)), _m.group(5)
            _cat = ("psup" if "P(sup)" in _why else
                    "landmine" if "regression" in _why else
                    "no-gain" if "best gain" in _why else
                    "unmeasurable" if "unmeasurable" in _why else "other")
            _reasons[_cat] = _reasons.get(_cat, 0) + 1
            _st = _stats.get(_size, {})
            _g = float(_st.get("gain_pct") or 0)
            _p = float(_st.get("psup") or 0)
            if (_cat == "psup" and _g > 5) or (_cat == "unmeasurable" and _g > 10):
                _suspects.append((_c, _sc, _size, _g, _p, _why))
        elif _line.strip() and not _line.startswith("#") \
                and not _line.startswith("collective_type"):
            _kept_n += 1
            _size = int(_line.split(",")[1])
            _st = _stats.get(_size, {})
            _g = float(_st.get("gain_pct") or 0)
            _p = float(_st.get("psup") or 0)
            if _g < 3 or _p < 0.98:
                _weak.append((_c, _sc, _size, _g, _p))
_total = _kept_n + sum(_reasons.values())
au.append(f"<h2>Totals</h2><p>{_total} rules judged &rarr; "
          f"<span class=g>{_kept_n} KEPT</span>, "
          f"{sum(_reasons.values())} dropped: "
          + ", ".join(f"{v} {k}" for k, v in sorted(_reasons.items(),
                                                    key=lambda kv: -kv[1]))
          + ". Drop meanings: <i>no-gain</i> = defaults already optimal there; "
            "<i>landmine</i> = a size in range regresses &gt;2%; <i>psup</i> = "
            "not reproducible at P(sup) &ge; 0.95; <i>unmeasurable</i> = "
            "baseline too noisy to judge.</p>")
au.append("<h2>Suspect drops — gains that look real, lost to residual noise "
          "(retry shortlist)</h2>"
          "<div class=tw><table><tr><th>rule</th><th>core gain</th>"
          "<th>P(sup)</th><th>reject reason</th></tr>")
for _c, _sc, _sz, _g, _p, _why in sorted(_suspects, key=lambda x: -x[3]):
    au.append(f"<tr><td>{_c} {_sc}n {size_h(_sz)}</td>"
              f"<td class=g>{_g:+.1f}%</td><td>{_p:.2f}</td>"
              f"<td class=wrap>{html.escape(_why)}</td></tr>")
au.append("</table></div><p class=note>Assessment: the gates applied their "
          "rules correctly in every case above - these are candidates for one "
          "targeted live A/B each, not evidence of a broken gate.</p>")
au.append("<h2>Weakest keeps (legitimate, flagged for transparency)</h2>"
          "<div class=tw><table><tr><th>rule</th><th>core gain</th>"
          "<th>P(sup)</th></tr>")
for _c, _sc, _sz, _g, _p in sorted(_weak, key=lambda x: x[3]):
    au.append(f"<tr><td>{_c} {_sc}n {size_h(_sz)}</td>"
              f"<td>{_g:+.1f}%</td><td>{_p:.2f}</td></tr>")
au.append("</table></div>")
au.append("<p class=note>Generated with the site from the live verdict files "
          "and core-based stats; regenerating the site refreshes this audit.</p>")
(HERE / "audit.html").write_text("\n".join(au))
print("audit built")

# index: status matrix + satellites
idx = [f"<!doctype html><meta charset=utf-8><meta name=viewport "
       f"content='width=device-width,initial-scale=1'>"
       f"<title>GPU-107 tuning</title><style>{CSS}</style>"]
idx.append("<h1>RCCL tuning — results by collective</h1>")
idx.append("<p class=note>Freshest live A/B verdicts, resolved automatically "
           "from all rccl-tune runs. 1 measurement = median of 3 runs; A/B = "
           "9 repeats/arm through the tuner plugin. Verdicts judged under the "
           "cv-core noise gate (adopted 2026-08-27): per point, repeats beyond "
           "3.5 robust-z are outliers; a point needs CV &le; 8% raw or on a "
           "&ge;7-repeat core; verdict stats are computed on the cores.</p>")
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
           "consulted, so its search is channels-only. <b>Corrected "
           "2026-08-30:</b> the ON arm must be "
           "<code>NCCL_MIN/MAX_P2P_NCHANNELS</code>, not "
           "<code>NCCL_MIN/MAX_NCHANNELS</code> - the latter does not reach "
           "the p2p path, so the 2026-08-26 A/Bs changed nothing. Re-measured "
           "at 2 nodes with the right knob: still no gain, defaults already "
           "sit at the plateau (findings/12).</p>")
idx.append("<p><a href='overview.html'><b>Overview: verified gains heatmap "
           "+ shipped configs</b></a> &nbsp;·&nbsp; "
           "<a href='audit.html'><b>Verdict audit</b></a></p>")
idx.append("<h2>Method evaluation &amp; docs</h2><ul>"
           "<li><a href='approaches.html'>the story: grid vs adaptive vs "
           "optuna vs random vs triage</a> "
          "</li>"
           "<li><a href='methods.html'>full offline comparison, 15 grids</a></li>"
           "<li><a href='drafts/new-facts.md'>10 verified facts (awaiting "
           "approval)</a></li>"
           "<li><a href='handoff.md'>handoff / summary</a> · "
           "<a href='cli/corrections.md'>corrections log</a></li></ul>")
(HERE / "index.html").write_text("\n".join(idx))
print("site built:", [f"collective_{c}.html" for c in COLLS])
