#!/usr/bin/env python3
"""CLI-run results pages: one pilot-style page per rccl-tune results dir.

Scans results-tuning/*-{pilot,stage*}/ for per-scale runs (adaptive_report.json)
and renders each dir's page: search cost vs grid, config before/after the A/B
(colored), final shipped config, per-size table with EXECUTED values from the
RCCL debug logs, raw A/B runs collapsible. Dark theme.
"""
import csv, glob, html, json, re
from pathlib import Path

HERE = Path(__file__).parent
RT = HERE.parent
CSS = """body{font-family:-apple-system,system-ui,sans-serif;margin:12px;max-width:980px;
background:#14161a;color:#d8dbe0}
table{border-collapse:collapse;font-size:12.5px;width:100%;display:block;overflow-x:auto}
th,td{border:1px solid #34383f;padding:3px 6px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td.wrap{white-space:normal;max-width:220px;text-align:left}
h1{font-size:20px;color:#f0f2f5}h2{font-size:16px;color:#e6e9ee}h3{font-size:14px;color:#e6e9ee}
.g{color:#4ec96a;font-weight:600}.b{color:#ff6b6b}.n{color:#8a8f98}
.v{background:#173321}.x{background:#331a1a}
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


def find_one(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(str(p)))
        if hits:
            return Path(hits[-1])
    return None


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


def stats_for(d, tag, nodes):
    stem = tag.replace("all_reduce", "allreduce").replace(
        "all_gather", "allgather").replace("reduce_scatter", "reducescatter")
    f = find_one(d / "ab_out*" / f"{stem}*" / "per_size_stats.csv")
    out = {}
    if f:
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


def render_dir(d):
    tags = sorted(p.parent.name for p in d.glob("*/adaptive_report.json"))
    if not tags:
        return None
    h = [f"<!doctype html><meta charset=utf-8><meta name=viewport "
         f"content='width=device-width,initial-scale=1'>"
         f"<title>{d.name}</title><style>{CSS}</style>"]
    h.append(f"<h1>{d.name} — adaptive end-to-end results</h1>")
    h.append("<p class=note>'executed' columns come from RCCL's own debug-log "
             "record (channel{Lo..Hi}) - what actually ran, both arms. "
             "'(trimmed)' = RCCL used fewer channels than the rule applied. "
             "Verdicts: per size, P(sup) &ge; 0.95 over the interleaved pairs, "
             "gain &gt; 2%, no regression.</p>")
    for tag in tags:
        m = re.match(r"(.+)_(\d)n$", tag)
        if not m:
            continue
        nodes = int(m.group(2))
        rep = json.load(open(d / tag / "adaptive_report.json"))
        vfile = find_one(d / f"{tag}_persize.validated.csv",
                         d / f"{tag}.validated.csv")
        conf = find_one(d / f"{tag}_persize.conf", d / f"{tag}.conf")
        final = find_one(d / f"{tag}.final.conf")
        st = stats_for(d, tag, nodes)
        h.append(f"<h2>{tag.replace('_', ' ', 1)}</h2>")
        cost = ""
        if "full_runs" in rep:
            cost = (f"search: <b>{rep['runs']} runs</b> vs full grid "
                    f"{rep['full_runs']} (<b>&minus;{rep['saving_pct']:.0f}%</b>), ")
        kept, dropped = rules(vfile) if vfile else ([], [])
        h.append(f"<p class=note>{cost}{rep.get('configs', '?')} configs "
                 f"measured, {rep.get('search_wall_s', 0) / 60:.0f} min. "
                 f"A/B: <b class=g>{len(kept)} kept</b>, "
                 f"<b class=b>{len(dropped)} dropped</b>"
                 + ("" if vfile else " <b class=b>(no A/B verdict)</b>") + ".</p>")
        if conf and vfile:
            h.append("<h3>Config: search output &rarr; A/B verdicts</h3>")
            h.append(conf_block(conf, kept, dropped))
        if final:
            h.append("<h3>Final shipped config</h3>"
                     f"<pre class=keep>{html.escape(final.read_text().strip())}</pre>")
        if st:
            h.append("<h3>Per size</h3>")
            h.append("<table><tr><th>size</th><th>default executed</th>"
                     "<th>ours requested</th><th>ours executed</th>"
                     "<th>default med</th><th>ours med</th><th>gain</th>"
                     "<th>P(sup)</th><th>verdict</th></tr>")
            winners = {int(s): w["cfg"] for s, w in rep["winners"].items()}
            for s in sorted(winners):
                r = st.get(s)
                verdict, cls = "no rule", " "
                for raw, lo, hi in kept:
                    if lo <= s <= hi:
                        verdict, cls = "KEPT", " class=v"
                for raw, lo, hi, why in dropped:
                    if lo <= s <= hi:
                        verdict, cls = f"dropped: {short_why(why)}", " class=x"
                if not r:
                    continue
                gain = float(r["gain_pct"]) if r["gain_pct"] else 0.0
                gcls = "g" if gain > 2 else ("b" if gain < -2 else "n")
                cexec = r.get("cfg_exec", "")
                if r.get("ch_trimmed") == "yes":
                    cexec += " (trimmed)"
                h.append(f"<tr{cls}><td>{size_h(s)}</td>"
                         f"<td>{r.get('def_exec', '')}</td>"
                         f"<td>{winners[s]}</td><td>{cexec}</td>"
                         f"<td>{r['def_median']}</td><td>{r['cfg_median']}</td>"
                         f"<td class={gcls}>{gain:+.1f}%</td>"
                         f"<td>{float(r['psup']):.2f}</td>"
                         f"<td class=wrap>{html.escape(verdict)}</td></tr>")
            h.append("</table>")
            h.append("<details><summary>raw A/B runs (busbw per size, both arms)"
                     "</summary><table><tr><th>size</th><th>arm</th><th>runs</th>"
                     "<th>median</th></tr>")
            for s in sorted(st):
                r = st[s]
                for arm, key, med in (("default", "def_runs", r["def_median"]),
                                      ("config", "cfg_runs", r["cfg_median"])):
                    h.append(f"<tr><td>{size_h(s)}</td><td>{arm}</td>"
                             f"<td>{r[key].replace(';', ' ')}</td>"
                             f"<td><b>{med}</b></td></tr>")
            h.append("</table></details>")
    h.append("<p class=note><a href='../index.html'>main results site</a></p>")
    out = HERE / f"{d.name}.html"
    out.write_text("\n".join(h))
    return out.name


pages = []
for d in sorted(RT.glob("2026-*-*")):
    if d.is_dir() and re.search(r"-(pilot|stage\w*)$", d.name):
        p = render_dir(d)
        if p:
            pages.append(p)
# index for the CLI pages + keep pilot.html as alias of the pilot dir page
idx = [f"<!doctype html><meta charset=utf-8><meta name=viewport "
       f"content='width=device-width,initial-scale=1'>"
       f"<title>rccl-tune runs</title><style>{CSS}</style>",
       "<h1>rccl-tune runs</h1><ul>"]
for p in pages:
    idx.append(f"<li><a href='{p}'>{p[:-5]}</a></li>")
idx.append("</ul><p class=note><a href='../index.html'>main results site</a></p>")
(HERE / "index.html").write_text("\n".join(idx))
pilot = [p for p in pages if p.endswith("-pilot.html")]
if pilot:
    (HERE / "pilot.html").write_text(
        (HERE / pilot[0]).read_text().replace("<h1>", "<h1>", 1))
print("pages:", pages)
