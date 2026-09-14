#!/usr/bin/env python3
"""bigmsg_2026-09-14.html - campaign A (rccl-tests all_reduce, 1 node, 128M..2G, RCCL 2.30.4).
Reads results-tuning/2026-09-14-1-bigmsg/runs_busbw.csv (fetched from node 2: one row per run x size, requested and
executed algo/proto/channels, busbw_ip) and, when present, the A/B outputs (ab_out/*/per_size_stats.csv, *.validated.csv).
Regenerate any time; the page states what is pending."""
import csv, html, statistics, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RUN = HERE.parent / (sys.argv[1] if len(sys.argv) > 1 else "2026-09-14-1-bigmsg")
OUT = HERE / (sys.argv[2] if len(sys.argv) > 2 else "bigmsg_2026-09-14.html")
SIZES = [134217728, 268435456, 536870912, 1073741824, 2147483648]
SZ = {134217728: "128M", 268435456: "256M", 536870912: "512M", 1073741824: "1G", 2147483648: "2G"}
CSS = """body{background:#0f1216;color:#d6dbe3;font:14px/1.5 system-ui,sans-serif;margin:0;padding:18px 26px 50px}
h1{font-size:21px;margin:0 0 6px}h2{font-size:16px;color:#5ab0ff;border-bottom:1px solid #2a323d;padding-bottom:4px;margin:30px 0 8px}
p{max-width:1100px;margin:6px 0}.note{color:#8b95a3;font-size:13px}code{background:#1d242e;padding:1px 5px;border-radius:4px;font-size:12.5px}
table{border-collapse:collapse;font-size:13px;margin:8px 0}th,td{border:1px solid #2a323d;padding:4px 9px;text-align:right}th{background:#1b222b}
td:first-child,th:first-child{text-align:left}.tw{overflow-x:auto;max-width:1150px}
tr.def td{background:#1d3a2a;font-weight:600}tr.best td{background:#2b3a55}.g{color:#5fd38a}.b{color:#ff7b72}.n{color:#8b95a3}
details{margin:8px 0}summary{cursor:pointer;color:#5ab0ff}svg text{font:12px system-ui}"""


def med(xs):
    return statistics.median(xs) if xs else None


def main():
    rows = list(csv.DictReader(open(RUN / "runs_busbw.csv")))
    # group: (req_algo, req_proto, req_ch) -> size -> [busbw]; executed cfg per group
    g = defaultdict(lambda: defaultdict(list)); ex = {}; nrun = defaultdict(set)
    for r in rows:
        k = (r["requested_algo"], r["requested_proto"], r["requested_ch"])
        g[k][int(r["size_bytes"])].append(float(r["busbw_ip"]))
        ex.setdefault(k, set()).add(f"{r['algo']}/{r['proto']}/{r['nchannels']}")
        nrun[k].add(r["run"])
    dkey = ("default", "default", "default")
    dmed = {s: med(g[dkey][s]) for s in SIZES} if dkey in g else {}

    def chsort(k):
        return (k[0] != "default", k[0], k[1], int(k[2]) if k[2].isdigit() else 999)
    keys = sorted(g, key=chsort)
    best = max((k for k in keys if k != dkey), key=lambda k: med(g[k][SIZES[-1]]) or 0)

    page = [f"<!doctype html><meta charset=utf-8><title>bigmsg 2026-09-14</title><style>{CSS}</style>",
            "<p><a href='index.html'>&larr; index</a> · <a href='dda_2026-09-14.html'>DDA explainer</a> · <a href='inmodel_rocm10_2026-09-10.html'>in-model rocm10</a></p>",
            "<h1>Campaign A: does any algo/proto/channels beat RCCL's default above 64 MiB? (all_reduce, 1 node, RCCL 2.30.4)</h1>",
            "<p class=note>2026-09-14, node amd-mi355x-2 (job 21245), image <code>lmsysorg/sglang:v0.5.19-rocm10-mi35x</code> (RCCL 2.30.4), container mode, "
            "<code>NCCL_MIN_NCHANNELS</code> removed, <code>-n 20 -w 5 -c 1 -A 1</code>, INFO log per run, 3 repeats per cell (median shown). "
            f"Sizes 128M..2G, above the DDA threshold (64 MiB inclusive), so every run is on the classic ring/tree path the tuner steers. "
            f"Data: <code>results-tuning/{RUN.name}/runs_busbw.csv</code> ({len(rows)} rows, {len(set(r['run'] for r in rows))} runs). "
            "Search = the validated racing policy: all combos at the anchor channel counts 32 and 48, then only combos within 15% of the leader get 56, 64, 112, 224.</p>"]
    # ---- table 1
    page.append("<h2>1. Median busbw (GB/s) per requested cell and size, and % vs the RCCL default</h2>")
    page.append("<div class=tw><table><tr><th>requested algo/proto/channels</th><th>executed (from -A 1)</th><th>runs</th>" + "".join(f"<th>{SZ[s]}</th>" for s in SIZES) + "<th>vs default @2G</th></tr>")
    for k in keys:
        cls = " class=def" if k == dkey else (" class=best" if k == best else "")
        label = "RCCL default (no request)" if k == dkey else f"{k[0]}/{k[1]}/{k[2]}"
        cells = ""
        for s in SIZES:
            m = med(g[k][s])
            if m is None: cells += "<td class=n>-</td>"; continue
            d = f" <span class=n>({(m / dmed[s] - 1) * 100:+.0f}%)</span>" if dmed.get(s) and k != dkey else ""
            cells += f"<td>{m:.0f}{d}</td>"
        v = ""
        if dmed.get(SIZES[-1]) and k != dkey:
            pct = (med(g[k][SIZES[-1]]) / dmed[SIZES[-1]] - 1) * 100
            v = f"<td class={'g' if pct > 2 else ('b' if pct < -2 else 'n')}>{pct:+.1f}%</td>"
        else:
            v = "<td class=n>reference</td>"
        exs = ", ".join(sorted(ex[k]))
        page.append(f"<tr{cls}><td>{label}</td><td>{html.escape(exs)}</td><td>{len(nrun[k])}</td>{cells}{v}</tr>")
    page.append("</table></div>")
    page.append("<p class=note>Green row = RCCL default; blue row = best forced cell. The default executes RING/SIMPLE/112 here (per <code>-A 1</code>; the channel column of -A 1 is a planned ceiling, "
                "and shows <code>-128</code> for the 224 request because it overflows above 127; the INFO logs hold the real channel span). "
                "Every LL128 request was executed as LL128: on RCCL 2.27.7 it was silently substituted to SIMPLE at 1 node (2026-08-03-2-combomatrix); on 2.30.4 it is honoured.</p>")
    # ---- chart at 2G and 128M
    page.append("<h2>2. Picture: median busbw at 128M and 2G per cell</h2>")
    ks = [k for k in keys]
    W = 1100; bar = 14; gap = 6; top = 30; left = 260
    hgt = top + len(ks) * (2 * bar + gap) + 20
    svg = [f"<svg viewBox='0 0 {W} {hgt}' width='100%' style='max-width:{W}px;background:#0b0e12;border:1px solid #2a323d;border-radius:6px'>"]
    mx = max(med(g[k][s]) or 0 for k in ks for s in (SIZES[0], SIZES[-1])) or 1
    scale = (W - left - 90) / mx
    svg.append(f"<text x='{left}' y='18' fill='#8b95a3'>light = 128M, dark = 2G; GB/s</text>")
    for i, k in enumerate(ks):
        y = top + i * (2 * bar + gap)
        label = "default" if k == dkey else f"{k[0]}/{k[1]}/{k[2]}"
        svg.append(f"<text x='{left - 8}' y='{y + bar + 4}' text-anchor='end' fill='#cfd3d9'>{html.escape(label)}</text>")
        for j, s in enumerate((SIZES[0], SIZES[-1])):
            m = med(g[k][s]) or 0
            col = ("#22c55e" if k == dkey else "#5ab0ff") if j == 1 else ("#86efac" if k == dkey else "#93c5fd")
            svg.append(f"<rect x='{left}' y='{y + j * bar}' width='{m * scale:.1f}' height='{bar - 1}' fill='{col}'/>")
            svg.append(f"<text x='{left + m * scale + 4:.1f}' y='{y + j * bar + bar - 3}' fill='#cfd3d9' font-size='11'>{m:.0f}</text>")
    svg.append("</svg>")
    page.append("".join(svg))
    # ---- A/B
    page.append("<h2>3. A/B verdict (validate_tuner_config.py, 9 repeats per arm, P(sup) &ge; 0.95 and &gt; 2% to keep a size)</h2>")
    conf = RUN / "all_reduce_1n.conf"
    if conf.exists():
        page.append(f"<p>Conf sent to the A/B (best cell per size from the search):</p><pre style='background:#0b0e12;border:1px solid #2a323d;padding:8px;font-size:12.5px'>{html.escape(conf.read_text().strip())}</pre>")
    stats = list(RUN.glob("ab_out/*/per_size_stats.csv")) + list(RUN.glob("*/per_size_stats.csv"))
    validated = list(RUN.glob("*.validated.csv"))
    if stats:
        for f in stats:
            rs = list(csv.DictReader(open(f)))
            if not rs: continue
            cols = [c for c in rs[0].keys() if c]
            page.append(f"<p class=note>{html.escape(str(f.relative_to(RUN)))}</p><div class=tw><table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>")
            for r in rs:
                page.append("<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols) + "</tr>")
            page.append("</table></div>")
    else:
        page.append("<p class=note>A/B running or not yet fetched: this section fills in when <code>per_size_stats.csv</code> / <code>*.validated.csv</code> land in the run dir.</p>")
    if validated:
        page.append("<p>Validated rules:</p><pre style='background:#0b0e12;border:1px solid #2a323d;padding:8px;font-size:12.5px'>" + html.escape("\n".join(f.read_text().strip() for f in validated)) + "</pre>")
    # ---- raw
    page.append("<details><summary>4. Raw runs (every run, every size)</summary><div class=tw><table><tr><th>run</th><th>requested</th><th>executed</th>" + "".join(f"<th>{SZ[s]}</th>" for s in SIZES) + "</tr>")
    byrun = defaultdict(dict); meta = {}
    for r in rows:
        byrun[r["run"]][int(r["size_bytes"])] = float(r["busbw_ip"])
        meta[r["run"]] = (f"{r['requested_algo']}/{r['requested_proto']}/{r['requested_ch']}", f"{r['algo']}/{r['proto']}/{r['nchannels']}")
    for run in sorted(byrun):
        page.append(f"<tr><td>{run}</td><td>{meta[run][0]}</td><td>{meta[run][1]}</td>" + "".join(f"<td>{byrun[run].get(s, float('nan')):.0f}</td>" for s in SIZES) + "</tr>")
    page.append("</table></div></details>")
    OUT.write_text("\n".join(page))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
