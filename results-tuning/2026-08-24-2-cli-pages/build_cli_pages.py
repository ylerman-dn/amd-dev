#!/usr/bin/env python3
"""Pilot results page for the rccl-tune CLI effort (dedicated folder)."""
import csv, html, json, re, statistics
from pathlib import Path

HERE = Path(__file__).parent
RT = HERE.parent
PILOT = RT / "2026-08-24-1-pilot"
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
        return f"landmine {m.group(1)}%"
    m = re.search(r"best gain only ([+-][\d.]+)%", why)
    if m:
        return f"no gain ({m.group(1)}%)"
    m = re.search(r"P\(sup\) only ([\d.]+)", why)
    if m:
        return f"not reproducible (P {m.group(1)})"
    return why[:40]


def rules(tag):
    kept, dropped = [], []
    for line in open(PILOT / f"{tag}_persize.validated.csv"):
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


def stats(statfile, nodes):
    out = {}
    with open(statfile) as f:
        for row in csv.DictReader(f):
            if int(row["nodes"]) == nodes:
                out[int(row["size_bytes"])] = row
    return out


def default_combo(nodes):
    """RCCL's own default choice per size, from the grid-era default rows
    (same RCCL build; grid nodes differ from the pilot's - combo choice is a
    build/topology decision, stable per scale)."""
    src = RT / f"2026-08-04-{ {1:1,2:4}[nodes] }-sweep{nodes}node/merged.csv".replace(" ", "")
    rows = {}
    with open(src) as f:
        for row in csv.DictReader(f):
            if (row.get("requested_algo") or "").strip():
                continue
            rows.setdefault(int(row["size_bytes"]), []).append(
                (row["algo"], row["proto"], row["nchannels"]))
    out = {}
    for sz, v in rows.items():
        a, p, ch = max(set(v), key=v.count)
        out[sz] = f"{a}/{p} ch~{ch}"
    return out


def conf_block(tag, kept, dropped):
    """The search's config with the A/B fate of every rule colored in."""
    kept_raw = {k[0] for k in kept}
    lines = []
    for line in open(PILOT / f"{tag}_persize.conf"):
        line = line.rstrip()
        if line.startswith("collective_type"):
            lines.append(f"<span class=n>{html.escape(line)}</span>")
            continue
        matched = False
        for raw, lo, hi, why in dropped:
            if line == raw:
                lines.append(f"<span class=drop>{html.escape(line)}</span>"
                             f"  <span class=why>&larr; {html.escape(short_why(why))}</span>")
                matched = True
        if not matched:
            cls = "keep" if line in kept_raw else "n"
            mark = "  <span class=why>&larr; kept</span>" if line in kept_raw else ""
            lines.append(f"<span class={cls}>{html.escape(line)}</span>{mark}")
    return "<pre>" + "\n".join(lines) + "</pre>"


SCALES = [
    ("all_reduce_1n", 1, "ab_out3/allreduce_1n/per_size_stats.csv", 81),
    ("all_reduce_2n", 2, "ab_out3/allreduce_2n_retry2/per_size_stats.csv", 162),
]

h = [f"<!doctype html><meta name=viewport content='width=device-width,"
     f"initial-scale=1'><title>rccl-tune pilot</title><style>{CSS}</style>"]
h.append("<h1>rccl-tune pilot — all_reduce, adaptive end-to-end</h1>")
h.append("<p class=note>One command: <code>rccl-tune run --collectives "
         "all_reduce --scales 1,2 --repeats 9</code>. Nodes {5,8}, "
         "2026-08-24, jobs 20723+20725. book &rarr; adaptive search &rarr; "
         "config &rarr; plugin A/B (9 repeats/arm) &rarr; verdicts.</p>")

for tag, nodes, statfile, grid_runs in SCALES:
    rep = json.load(open(PILOT / tag / "adaptive_report.json"))
    kept, dropped = rules(tag)
    st = stats(PILOT / statfile, nodes)
    dc = default_combo(nodes)
    h.append(f"<h2>{tag.replace('_', ' ', 1)}</h2>")
    h.append(f"<p class=note>search: <b>{rep['runs']} runs</b> vs full grid "
             f"{grid_runs} (<b>&minus;{(1 - rep['runs'] / grid_runs) * 100:.0f}%"
             f"</b>), {rep['configs']} configs measured, "
             f"{rep['search_wall_s'] / 60:.0f} min search. A/B: "
             f"<b class=g>{len(kept)} rules kept</b>, "
             f"<b class=b>{len(dropped)} dropped</b>.</p>")

    h.append("<h3>Config: before A/B &rarr; what ships</h3>")
    h.append("<p class=note>green = survived the A/B; struck red = the A/B "
             "killed it (reason inline). Sizes with no rule keep RCCL's "
             "default.</p>")
    h.append(conf_block(tag, kept, dropped))
    final = (PILOT / f"{tag}.final.conf").read_text().strip()
    h.append("<h3>Final shipped config (survivors, merged)</h3>"
             f"<pre class=keep>{html.escape(final)}</pre>")

    h.append("<h3>Per size</h3>")
    h.append("<table><tr><th>size</th><th>RCCL default combo</th>"
             "<th>our config</th><th>default med</th><th>ours med</th>"
             "<th>gain</th><th>P(sup)</th><th>verdict</th></tr>")
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
        dcombo = dc.get(s, "?")
        if r:
            gain = float(r["gain_pct"])
            gcls = "g" if gain > 2 else ("b" if gain < -2 else "n")
            h.append(f"<tr{cls}><td>{size_h(s)}</td><td>{dcombo}</td>"
                     f"<td>{winners[s]}</td><td>{r['def_median']}</td>"
                     f"<td>{r['cfg_median']}</td>"
                     f"<td class={gcls}>{gain:+.1f}%</td><td>{float(r['psup']):.2f}</td>"
                     f"<td class=wrap>{html.escape(verdict)}</td></tr>")
    h.append("</table>")

    # raw runs, collapsed - the medians' full evidence
    h.append("<details><summary>raw A/B runs (9 per arm, busbw per size)"
             "</summary><table><tr><th>size</th><th>arm</th>"
             + "".join(f"<th>r{i}</th>" for i in range(1, 10))
             + "<th>median</th></tr>")
    for s in sorted(st):
        r = st[s]
        for arm, key, med in (("default", "def_runs", r["def_median"]),
                              ("config", "cfg_runs", r["cfg_median"])):
            vals = r[key].split(";")
            h.append(f"<tr><td>{size_h(s)}</td><td>{arm}</td>"
                     + "".join(f"<td>{v}</td>" for v in vals)
                     + f"<td><b>{med}</b></td></tr>")
    h.append("</table></details>")

h.append("<h2>Attempt history</h2>")
h.append("<pre>pilot 1: failed fast - node 5 python lacks tabulate; empty config emitted silently.\n"
         "         fixed: exec-node env probe + hard-fail on zero winners.\n"
         "pilot 2: 1n search 72 runs, 2n search 117 runs, 1n A/B verdict first try.\n"
         "         2n A/B: 6 preflight refusals across the day (single-run collapses\n"
         "         at 1M-256M, exporter signature), verdict on attempt 7 overall.\n"
         "redo   : per-size rules (verdict granularity = measurement granularity).\n"
         "         1n: 9/18 sizes kept, all P(sup)=1.00 - incl 64K alone at 1.00.\n"
         "         2n: 2/18 kept (32M +7.2%, 128M +2.4%); yesterday's 262K win did\n"
         "         not reproduce today and correctly did not ship.</pre>")
h.append("<p class=note><a href='../index.html'>main results site</a></p>")
(HERE / "pilot.html").write_text("\n".join(h))
print("wrote pilot.html")
