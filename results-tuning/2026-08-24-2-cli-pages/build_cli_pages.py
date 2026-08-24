#!/usr/bin/env python3
"""Pilot results pages for the rccl-tune CLI effort (separate, clean folder).

Everything here is from the 2026-08-24 pilot: the first ADAPTIVE-SOURCED
end-to-end run (search -> config -> plugin A/B), all_reduce 1n+2n on {5,8}.
"""
import csv, html, json, re
from pathlib import Path

HERE = Path(__file__).parent
PILOT = HERE.parent / "2026-08-24-1-pilot"
CSS = """body{font-family:-apple-system,system-ui,sans-serif;margin:12px;max-width:980px;
background:#14161a;color:#d8dbe0}
table{border-collapse:collapse;font-size:13px;width:100%;display:block;overflow-x:auto}
th,td{border:1px solid #34383f;padding:4px 6px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
tr:nth-child(even){background:#1b1e24}
h1{font-size:20px;color:#f0f2f5}h2{font-size:16px;color:#e6e9ee}h3{font-size:14px}
.g{color:#4ec96a;font-weight:600}.b{color:#ff6b6b}.n{color:#8a8f98}
.v{background:#173321}.x{background:#331a1a}
.note{color:#9aa0a8;font-size:13px}
a{color:#6ea8ff;text-decoration:none}
pre{background:#1b1e24;padding:8px;font-size:12px;overflow-x:auto;border-radius:6px}"""


def size_h(b):
    for u, d in (("M", 1 << 20), ("K", 1 << 10)):
        if b >= d:
            v = b / d
            return f"{v:.0f}{u}" if v == int(v) else f"{v:.1f}{u}"
    return str(b)


def rules(tag):
    """(lo, hi) -> (verdict, detail) from the validated csv."""
    kept, dropped = [], []
    with open(PILOT / f"{tag}.validated.csv") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"# dropped: \w+,(\d+),(\d+),.*?\((.*)\)$", line)
            if m:
                dropped.append((int(m.group(1)), int(m.group(2)), m.group(3)))
            elif line and not line.startswith("#") and \
                    not line.startswith("collective_type"):
                p = line.split(",")
                kept.append((int(p[1]), int(p[2])))
    return kept, dropped


def stats(statfile, nodes):
    out = {}
    with open(statfile) as f:
        for row in csv.DictReader(f):
            if int(row["nodes"]) != nodes:
                continue
            out[int(row["size_bytes"])] = row
    return out


SCALES = [
    ("all_reduce_1n", 1, "ab_out/allreduce_1n/per_size_stats.csv", 81),
    ("all_reduce_2n", 2, "ab_out2/allreduce_2n_retry4/per_size_stats.csv", 162),
]

h = [f"<!doctype html><meta name=viewport content='width=device-width,"
     f"initial-scale=1'><title>rccl-tune pilot</title><style>{CSS}</style>"]
h.append("<h1>rccl-tune pilot — all_reduce, adaptive end-to-end</h1>")
h.append("<p class=note>First full-chain run of the CLI: book &rarr; adaptive "
         "search &rarr; config &rarr; plugin A/B (9 repeats) &rarr; verdicts. "
         "Nodes {5,8}, 2026-08-24, jobs 20723 + 20725. Search sourced the "
         "configs (not the old grids). Spread = min-max over the 9 runs of "
         "each arm.</p>")

for tag, nodes, statfile, grid_runs in SCALES:
    rep = json.load(open(PILOT / tag / "adaptive_report.json"))
    kept, dropped = rules(tag)
    st = stats(PILOT / statfile, nodes)
    h.append(f"<h2>{tag.replace('_', ' ', 1)} </h2>")
    h.append(f"<p class=note>search: <b>{rep['runs']} runs</b> vs full grid "
             f"{grid_runs} (<b>&minus;{(1 - rep['runs'] / grid_runs) * 100:.0f}%"
             f"</b>), {rep['configs']} configs measured, "
             f"{rep['search_wall_s'] / 60:.0f} min. A/B verdict: "
             f"<b>{len(kept)} rules kept, {len(dropped)} dropped</b>.</p>")
    h.append("<table><tr><th>size</th><th>winner (search)</th>"
             "<th>default median (A/B)</th><th>config median (A/B)</th>"
             "<th>gain</th><th>P(sup)</th><th>spread def / cfg</th>"
             "<th>rule verdict</th></tr>")
    winners = {int(s): w["cfg"] for s, w in rep["winners"].items()}
    for s in sorted(winners):
        r = st.get(s)
        verdict, cls = "no rule (tie with default)", " class=n"
        for lo, hi in kept:
            if lo <= s <= hi:
                verdict, cls = "KEPT", " class=v"
        for lo, hi, why in dropped:
            if lo <= s <= hi:
                verdict, cls = f"dropped: {why}", " class=x"
        if r:
            gain = float(r["gain_pct"])
            gcls = "g" if gain > 2 else ("b" if gain < -2 else "n")
            h.append(f"<tr{cls}><td>{size_h(s)}</td><td>{winners[s]}</td>"
                     f"<td>{r['def_median']}</td><td>{r['cfg_median']}</td>"
                     f"<td class={gcls}>{gain:+.1f}%</td><td>{r['psup']}</td>"
                     f"<td class=n>{r['def_min']}-{r['def_max']} / "
                     f"{r['cfg_min']}-{r['cfg_max']}</td>"
                     f"<td>{html.escape(str(verdict))}</td></tr>")
        else:
            h.append(f"<tr{cls}><td>{size_h(s)}</td><td>{winners[s]}</td>"
                     f"<td colspan=5 class=n>no A/B data</td>"
                     f"<td>{html.escape(str(verdict))}</td></tr>")
    h.append("</table>")

h.append("<h2>Attempt history (honesty section)</h2>")
h.append("<pre>pilot 1: failed fast - node 5 python lacks tabulate; empty "
         "config emitted silently.\n         Fixed: exec-node env probe + "
         "hard-fail on zero winners.\npilot 2: 1n search 72 runs OK; 2n "
         "search 117 runs OK; 1n A/B verdict first try.\n         2n A/B: 3 "
         "preflight refusals (single-run collapses at 1M-256M, exporter "
         "signature),\n         focused burst: 3 more refusals, then verdict "
         "on attempt 4.</pre>")
h.append("<p class=note><a href='../2026-08-23-1-pages/index.html'>"
         "main results site</a></p>")
(HERE / "pilot.html").write_text("\n".join(h))
print("wrote pilot.html")
