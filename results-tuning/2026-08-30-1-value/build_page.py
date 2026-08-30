#!/usr/bin/env python3
"""Build a self-contained results page for the 2026-08-30 value runs.

Reads the rccl-tests CSVs in leaf-ab/ and alltoall-2n/, computes mean and median busbw
per (arm, size) plus percent change against each arm's reference, and writes results.html
with no external assets. Stdlib only — the dev VM has no pandas.
"""
import csv
import glob
import html
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

# rccl-tests writes 13 header names over 14-field rows, so DictReader misaligns.
I_SIZE, I_INPLACE, I_BUSBW = 6, 9, 12


def load(directory, arm):
    by_size = {}
    for path in sorted(glob.glob(os.path.join(directory, f"{arm}_rep*.csv"))):
        with open(path, newline="") as fh:
            for row in csv.reader(fh):
                if len(row) < 14 or row[I_INPLACE].strip('"') != "0":
                    continue
                try:
                    by_size.setdefault(int(row[I_SIZE]), []).append(float(row[I_BUSBW]))
                except ValueError:
                    continue
    return by_size


def human(size):
    for unit, div in (("M", 1 << 20), ("K", 1 << 10)):
        if size >= div:
            return f"{size // div}{unit}"
    return str(size)


def pct_class(value):
    if value is None:
        return ""
    if value >= 5:
        return "big-up"
    if value >= 1.5:
        return "up"
    if value <= -5:
        return "big-down"
    if value <= -1.5:
        return "down"
    return "flat"


def table(directory, arms, ref, caption, note, ruled=()):
    data = {a: load(directory, a) for a in arms}
    sizes = sorted({s for a in arms for s in data[a]})
    head = "".join(f"<th colspan='2'>{html.escape(a)}</th>" for a in arms)
    sub = "".join("<th class='sub'>mean</th><th class='sub'>median</th>" for _ in arms)
    dhead = "".join(f"<th>{html.escape(a)}</th>" for a in arms if a != ref)

    rows = []
    for size in sizes:
        ref_med = statistics.median(data[ref][size]) if size in data[ref] else None
        cells, deltas = [], []
        for arm in arms:
            vals = data[arm].get(size)
            if vals:
                cells.append(f"<td>{statistics.mean(vals):.2f}</td>"
                             f"<td class='med'>{statistics.median(vals):.2f}</td>")
            else:
                cells.append("<td>—</td><td class='med'>—</td>")
            if arm != ref:
                if vals and ref_med:
                    d = (statistics.median(vals) - ref_med) / ref_med * 100
                    deltas.append(f"<td class='{pct_class(d)}'>{d:+.1f}%</td>")
                else:
                    deltas.append("<td>—</td>")
        n = len(data[ref].get(size, []))
        mark = " <span class='rule-tag'>rule</span>" if size in ruled else ""
        rows.append(f"<tr><th class='size'>{human(size)}{mark}</th>"
                    f"<td class='n'>{n}</td>{''.join(cells)}{''.join(deltas)}</tr>")

    return f"""
<section>
<h3>{html.escape(caption)}</h3>
<p class="note">{note}</p>
<div class="scroll">
<table>
<thead>
<tr><th rowspan="2" class="size">size</th><th rowspan="2" class="n">n</th>{head}
    <th colspan="{len(arms)-1}">change vs {html.escape(ref)} (median)</th></tr>
<tr>{sub}{dhead}</tr>
</thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>
</section>"""


CSS = """
:root{--bg:#fbfaf8;--fg:#22201d;--muted:#6b6660;--line:#e3ded6;--card:#fff;
--up:#1a7f4b;--bigup:#0f5c34;--down:#b3402c;--bigdown:#8a2c1c;--accent:#8a5a2b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:40px 24px 72px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:21px;margin:44px 0 10px;padding-bottom:7px;border-bottom:2px solid var(--fg);
letter-spacing:-.01em}
h3{font-size:16px;margin:26px 0 4px}
.lede{color:var(--muted);margin:0 0 4px}
.meta{color:var(--muted);font-size:13px;margin:0 0 8px}
.note{color:var(--muted);font-size:13.5px;margin:0 0 12px;max-width:78ch}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:9px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13.5px}
th,td{padding:6px 10px;text-align:right;white-space:nowrap}
thead th{background:#f3efe8;font-weight:600;border-bottom:1px solid var(--line);
position:sticky;top:0}
thead th.sub{font-weight:500;color:var(--muted);font-size:12px}
th.size{text-align:left;font-weight:600}
td.n,th.n{color:var(--muted);font-size:12px}
td.med{font-weight:600}
tbody tr:nth-child(even){background:#faf8f4}
tbody tr:hover{background:#f2ede4}
.up{color:var(--up)}.big-up{color:var(--bigup);font-weight:700}
.down{color:var(--down)}.big-down{color:var(--bigdown);font-weight:700}
.flat{color:var(--muted)}
.rule-tag{background:var(--accent);color:#fff;font-size:10px;padding:1px 5px;
border-radius:9px;vertical-align:middle;letter-spacing:.03em}
.cards{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));margin:18px 0 6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:15px 17px}
.card h4{margin:0 0 5px;font-size:13px;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted)}
.card .big{font-size:25px;font-weight:700;letter-spacing:-.02em}
.card p{margin:5px 0 0;font-size:13px;color:var(--muted)}
code{background:#f0ebe2;padding:1px 5px;border-radius:4px;font-size:12.5px}
.foot{margin-top:52px;padding-top:16px;border-top:1px solid var(--line);
color:var(--muted);font-size:12.5px}
@media (prefers-color-scheme:dark){
:root:not([data-theme=light]){--bg:#16150f;--fg:#ece7dd;--muted:#9d968a;--line:#332f27;
--card:#1e1c15;--up:#5fd39a;--bigup:#7ee3b0;--down:#f08a72;--bigdown:#ff9d85;--accent:#c08a4a}
:root:not([data-theme=light]) thead th{background:#252219}
:root:not([data-theme=light]) tbody tr:nth-child(even){background:#1a1811}
:root:not([data-theme=light]) tbody tr:hover{background:#272319}
:root:not([data-theme=light]) code{background:#272319}
}
"""


def main():
    leaf = os.path.join(HERE, "leaf-ab")
    a2a = os.path.join(HERE, "alltoall-2n")

    body = []
    body.append("""
<h2>1 &nbsp;Leaf placement — does a tuning gain survive the spine?</h2>
<p class="note">Same-leaf <code>{3,5}</code> (both L1) versus cross-leaf <code>{3,9}</code>
(L1&rarr;L2, over the shared spine). Node 3 anchors both arms. 5 interleaved repeats per arm,
<code>all_reduce_perf -b 4K -e 512M -n 20 -w 5</code>. Every tuned run verified by
<code>Applied config</code> in 16/16 rank logs; every default run shows 0/16. The shipped 2-node
config carries rules at 256K and 32M only — those rows are tagged.</p>
<div class="cards">
<div class="card"><h4>Baseline shift, 16K</h4><div class="big big-down">-17.4%</div>
<p>cross-leaf slower; penalty decays with size and is gone above 8M</p></div>
<div class="card"><h4>32M rule, same leaf</h4><div class="big big-up">+9.5%</div>
<p>every tuned repeat above every default repeat</p></div>
<div class="card"><h4>32M rule, across spine</h4><div class="big big-up">+7.7%</div>
<p>gain survives the extra hops — no overlap between arms</p></div>
</div>""")

    body.append(table(
        leaf,
        ["same_def", "same_tun", "cross_def", "cross_tun"],
        "same_def",
        "All four arms against the same-leaf default",
        "Reference is <code>same_def</code>. So the <code>cross_def</code> column is the pure "
        "placement effect, and <code>same_tun</code> is the pure tuning effect.",
        ruled={262144, 33554432}))

    body.append(table(
        leaf, ["cross_def", "cross_tun"], "cross_def",
        "Tuning effect measured inside the cross-leaf arm",
        "Both arms crossed the spine, so placement cancels and only the tuner differs. "
        "The 32M rule holds at +7.7%. The 256K rule reads +3.7%, but repeat spread at that "
        "size is 7-12% and a no-rule 16K control swings +6% on noise alone, so it is not "
        "resolvable at 5 repeats.",
        ruled={262144, 33554432}))

    body.append("""
<h2>2 &nbsp;alltoall — the knob that was never turned</h2>
<p class="note">Earlier alltoall work drove <code>NCCL_MIN/MAX_NCHANNELS</code>, which bounds the
<em>collective</em> path. alltoall runs on the <em>p2p</em> path, controlled by
<code>NCCL_MIN/MAX_P2P_NCHANNELS</code> (<code>src/graph/paths.cc:982-983</code>). Those runs
changed nothing, so the "+0.00% median" they reported was a null edit. Redone here with the
correct variable, 5 repeats per arm on <code>{3,5}</code>.</p>
<div class="cards">
<div class="card"><h4>8 p2p channels, 256M</h4><div class="big big-down">-26.6%</div>
<p>the knob demonstrably moves busbw — the measurement is real now</p></div>
<div class="card"><h4>16 / 32 / 64 channels</h4><div class="big flat">~0%</div>
<p>default already sits at the plateau; only downward moves are available</p></div>
<div class="card"><h4>Tuner consultations</h4><div class="big flat">0</div>
<p>plugin loaded and holding rules; RCCL never asked it, for any size</p></div>
</div>""")

    body.append(table(
        a2a,
        ["def", "p2p8", "p2p16", "p2p32", "p2p64", "pivot", "plugdef", "plugpivot"],
        "def",
        "alltoall 2 nodes — all eight arms",
        "<code>p2pN</code> forces N p2p channels. <code>pivot</code> sets "
        "<code>RCCL_ALL_TO_ALL_PIVOT_ENABLE=1</code>, which would route alltoall through the "
        "collective path where the tuner lives — it did not engage on this topology. "
        "<code>plugdef</code> and <code>plugpivot</code> load the tuner plugin; both produced "
        "zero <code>Applied config</code> and zero <code>Config does not match</code> lines "
        "across the whole run, which is the log proof that RCCL never consults the tuner for "
        "alltoall. 2 of 40 runs died with the known intermittent "
        "<code>ionic_comp</code> error, so some cells rest on 4 repeats."))

    body.append("""
<h2>3 &nbsp;Inference — Llama-3.1-8B, TP=8, single node</h2>
<p class="note">SGLang v0.5.17, dense model so all_reduce is effectively the only collective.
<code>hidden_size</code> 4096 in bf16 puts the per-layer all_reduce at 8KB — inside the shipped
1-node rule band. ISL 512 / OSL 512, concurrency 32, 128 prompts.</p>
<div class="scroll">
<table>
<thead><tr><th class="size">arm</th><th>RCCL</th><th>all_reduce path</th>
<th>tok/s</th><th>TPOT ms</th><th>median ITL ms</th><th class="size">outcome</th></tr></thead>
<tbody>
<tr><th class="size">A</th><td>ours</td><td>SGLang custom kernel (default)</td>
<td class="med">687.35</td><td>14.09</td><td>13.97</td>
<td class="size">RCCL bypassed — only 6 AllReduce calls in the entire log</td></tr>
<tr><th class="size">B</th><td>ours</td><td>RCCL</td><td colspan="3" class="big-down">hang</td>
<td class="size">deterministic, twice, at p2p Send/Recv <code>opCount 989</code></td></tr>
<tr><th class="size">C</th><td>stock</td><td>RCCL</td>
<td class="med">668.22</td><td>15.69</td><td>15.61</td>
<td class="size">completes — so the hang is our build, not the SGLang path</td></tr>
<tr><th class="size">D</th><td>stock</td><td>RCCL + tuner</td><td colspan="3" class="flat">tuner never fires</td>
<td class="size">72 consultations, all at init; 0 across 12488 steady-state all_reduces</td></tr>
<tr><th class="size">E</th><td>stock</td><td>rccl-tests in the same container</td>
<td colspan="3" class="big-up">9 rules applied</td>
<td class="size">all 8 rule sizes incl. 8192 and 16384 — so the library is fine</td></tr>
</tbody></table></div>
<p class="note" style="margin-top:14px">Arm E is the one that matters: the same stock RCCL, in the
same container, <em>does</em> consult the tuner per collective. So arm D's silence is something in
SGLang's calling pattern, not a library limitation. The demo is reachable. Separately, arm A vs C
shows that routing all_reduce through RCCL at all costs <strong>2.8%</strong> throughput versus
SGLang's own kernel — a deficit any tuning must first make up.</p>""")

    page = f"""<title>GPU-107 value runs — 2026-08-30</title>
<style>{CSS}</style>
<div class="wrap">
<h1>Proving value — 2026-08-30</h1>
<p class="lede">Leaf placement, alltoall re-test, and a first inference attempt.</p>
<p class="meta">Allocation 20856 · nodes amd-mi355x-[2-3,5,9] · RCCL 2.17.9-develop:2e42aa8 ·
busbw in GB/s, out-of-place rows · medians in bold, percent changes computed on medians</p>
{''.join(body)}
<p class="foot">Raw CSVs, logs and per-task SUMMARY.md live beside this file in
<code>results-tuning/2026-08-30-1-value/</code>. Regenerate with
<code>python3 build_page.py</code>. Node 2 is excluded throughout: every MPI benchmark on
<code>{{3,2}}</code> died with <code>pmixp_server.c:1582: Cannot send message</code> while
<code>{{3,5}}</code> ran clean with identical environment.</p>
</div>
"""
    out = os.path.join(HERE, "results.html")
    with open(out, "w") as fh:
        fh.write(page)
    print(f"wrote {out} ({len(page)} bytes)")


if __name__ == "__main__":
    main()
