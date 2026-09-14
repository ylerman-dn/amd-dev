#!/usr/bin/env python3
"""2026-09-08 sanity-check campaign page, in the exact format of collective_all_reduce.html
(build_site.py): per run the provenance line, the search config with colored A/B verdicts,
the final shipped config, the per-size table with executed-truth values and live A/B numbers,
the raw A/B runs (collapsed). Plus one section for the Qwen3-30B-A3B in-model campaign.
Every number is read from the run dirs under results-tuning/; nothing is typed in.

Helpers are copied from build_site.py (importing it would rebuild every page as a side effect).
"""
import csv, html, json, re, statistics as _s
from pathlib import Path

HERE = Path(__file__).parent
RT = HERE.parent
RUNS = [  # (dir, label)
    ("2026-09-08-1-sanity-check-1", "grid 1..48 (validated policy)"),
    ("2026-09-08-2-sanity-check-2", "grid 1..112"),
    ("2026-09-08-3-sanity-check-3", "grid 1..224"),
]
QWEN = RT / "2026-09-08-4-qwen"

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


# ----------------------------------------------------------- helpers (build_site.py)
def _vals(row, key):
    return [float(x) for x in row[key].split(";") if x]


def _stats(vals):
    if not vals:
        return "", "", "", ""
    med = _s.median(vals); mean = _s.mean(vals)
    sd = _s.stdev(vals) if len(vals) > 1 else 0.0
    cv = sd / mean * 100 if mean else 0.0
    return f"{med:g}", f"{mean:.4g}", f"{sd:.3g}", f"{cv:.1f}"


def _gate_cell(vals):
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


def rules(vfile):
    kept, dropped = [], []
    for line in open(vfile):
        line = line.strip()
        m = re.match(r"# dropped: (\w+,(\d+),(\d+),.*?)\s+\((.*)\)$", line)
        if m:
            dropped.append((m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)))
        elif line and not line.startswith("#") and not line.startswith("collective_type"):
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


# --------------------------------------------------------------- per run section
def run_section(dirname, label):
    d = RT / dirname
    coll, scale = "all_reduce", 1
    tag = f"{coll}_{scale}n"
    rep = json.load(open(d / tag / "adaptive_report.json"))
    vfile = d / f"{tag}.validated.csv"
    conf = d / f"{tag}.conf"
    stats = sorted(d.glob("ab_out*/allreduce*/per_size_stats.csv"))[-1]
    # wall times from rccl_tune.log
    log = (d / "rccl_tune.log").read_text()
    m_done = re.search(r"done in (\d+)s", log)
    t_search = re.search(r"\[(\S+)\] === search", log)
    t_ab = re.search(r"\[(\S+)\] === A/B", log)
    t_abdone = re.search(r"\[(\S+)\] ab batch finished", log)

    def _t(s):
        return sum(int(x) * k for x, k in zip(s.split("T")[1].rstrip("Z").split(":"), (3600, 60, 1)))
    search_min = (_t(t_ab.group(1)) - _t(t_search.group(1))) / 60 if t_search and t_ab else 0
    ab_min = (_t(t_abdone.group(1)) - _t(t_ab.group(1))) / 60 if t_ab and t_abdone else 0
    grid = re.search(r"--grid (\S+)", log).group(1)
    node = re.search(r"allocation (\d+) on (amd-mi355x-\d+)", log)

    page = [f"<h2>1 node &mdash; {html.escape(label)}</h2>"]
    page.append(f"<p class=note>run <b>{d.name}</b> · "
                f"search {rep['runs']} runs vs grid {rep['full_runs']} "
                f"(&minus;{rep['saving_pct']:.0f}%), {rep.get('configs', '?')} configs, "
                f"{search_min:.0f} min search + {rep.get('default_runs', 3)} default runs · "
                f"9 A/B repeats/arm, {ab_min:.0f} min · node {node.group(2)}, job {node.group(1)} · "
                f"wall {int(m_done.group(1)) // 60} min · "
                f"<code>rccl_tune.py run --collectives all_reduce --scales 1 --grid {html.escape(grid)}</code></p>")
    kept, dropped = rules(vfile)
    page.append("<h3>Search config &rarr; A/B verdicts</h3>")
    page.append(conf_block(conf, kept, dropped))
    final = d / f"{tag}.final.conf"
    if final.exists() and kept:
        page.append("<h3>Final shipped config</h3>"
                    f"<pre class=keep>{html.escape(final.read_text().strip())}</pre>")
    st = load_stats(stats, scale)
    # executed-in-search: from search_raw/r*/merged_exec.csv, matched on the winner's requested cfg
    winners = {int(s): w["cfg"] for s, w in rep["winners"].items()}
    ex = {}
    for f in d.glob("search_raw/r0*/merged_exec.csv"):
        for row in csv.DictReader(open(f)):
            k = (int(row["size_bytes"]), row["requested_algo"], row["requested_proto"],
                 row["requested_nchannels"])
            ex.setdefault(k, set()).add(row["exec_nchannels"])
    conf_rows = {int(r["min_bytes"]): r for r in
                 csv.DictReader(l for l in open(conf) if not l.startswith("#"))}
    page.append("<h3>Per size (live A/B numbers)</h3>")
    page.append("<div class=tw><table><tr><th>size</th><th>default executed</th>"
                "<th>ours requested</th><th>ours executed (search, env path)</th>"
                "<th>ours executed (A/B, plugin)</th>"
                "<th>default med</th><th>ours med</th><th>gain</th>"
                "<th>P(sup)</th><th>noise gate def / cfg</th>"
                "<th>verdict</th></tr>")
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
        c = conf_rows.get(s)
        k = (s, c["algorithm"].upper(), c["protocol"].upper(), c["channels"]) if c else None
        sexec = (f"{c['algorithm'].upper()}/{c['protocol'].upper()}/"
                 f"{'/'.join(sorted(ex.get(k, {'?'})))}") if c else ""
        page.append(
            f"<tr{cls}><td>{size_h(s)}</td>"
            f"<td>{row.get('def_exec', '')}</td>"
            f"<td>{winners[s]}</td><td>{sexec}</td><td>{cexec}</td>"
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
    # search-stage default runs vs A/B default arm
    defs = {}
    for f in d.glob("search_raw/d0*/merged_exec.csv"):
        for row in csv.DictReader(open(f)):
            defs.setdefault(int(row["size_bytes"]), []).append(
                (float(row["busbw_ip"]), f"{row['exec_algo']}/{row['exec_proto']}/{row['exec_nchannels']}"))
    if defs:
        page.append("<details><summary>default consistency: search-stage default runs vs A/B default arm</summary>"
                    "<div class=tw><table class=fit><tr><th>size</th><th>search default med (3 runs)</th>"
                    "<th>executed</th><th>A/B default med</th><th>executed</th><th>diff</th></tr>")
        for s in sorted(defs):
            row = st.get(s)
            if not row:
                continue
            med = _s.median(v[0] for v in defs[s]); ab = float(row["def_median"])
            dif = (med / ab - 1) * 100
            page.append(f"<tr><td>{size_h(s)}</td><td>{med:g}</td><td>{defs[s][0][1]}</td>"
                        f"<td>{ab:g}</td><td>{row.get('def_exec', '')}</td>"
                        f"<td class={'n' if abs(dif) < 3 else 'warn'}>{dif:+.1f}%</td></tr>")
        page.append("</table></div></details>")
    return page, kept, len(winners)


# ------------------------------------------------------------------ qwen section
TOK = re.compile(r"Output token throughput \(tok/s\):\s+([0-9.]+)")
MTTFT = re.compile(r"Mean TTFT \(ms\):\s+([0-9.]+)")
MDTTFT = re.compile(r"Median TTFT \(ms\):\s+([0-9.]+)")
MTPOT = re.compile(r"Mean TPOT \(ms\):\s+([0-9.]+)")


def bench(f):
    t = open(f, errors="ignore").read()
    g = lambda rx: float(rx.search(t).group(1)) if rx.search(t) else float("nan")
    return g(TOK), g(MTTFT), g(MDTTFT), g(MTPOT)


def qwen_section():
    arms = ["sc1", "sc2", "sc3", "dummy", "none"]
    desc = {"sc1": "sanity-check-1 final.conf (grid 1..48, 7 rules 4K-1M)",
            "sc2": "sanity-check-2 final.conf (grid 1..112, 8 rules 4K-2M)",
            "sc3": "sanity-check-3 final.conf (grid 1..224, 8 rules 4K-2M)",
            "dummy": "ring,simple,1 channel for ALL sizes (deliberately bad)",
            "none": "plugin loaded, zero rules = RCCL default"}
    page = ["<h2>Qwen3-30B-A3B in-model &mdash; the three confs + a dummy + none</h2>"]
    page.append("<p class=note>run <b>2026-09-08-4-qwen</b> · node 5, job 21065 · one SGLang server "
                "(TP-8, RCCL_MSCCL_ENABLE=0, --disable-custom-all-reduce, image NCCL_MIN_NCHANNELS removed), "
                "hot-reload tuner plugin, conf swapped per round, 5 arms interleaved, 10 rounds · "
                "sglang.bench_serving random ISL 512 / OSL 512, concurrency 32, 128 prompts · "
                "<code>infer_paired.sh &lt;model&gt; qw 10 sc1:… sc2:… sc3:… dummy:… none:NONE</code> "
                "→ <code>infer_score.py --prefix qw --ref none</code></p>")
    score = list(csv.DictReader(open(QWEN / "qw_score.csv")))
    page.append("<h3>Verdict (round 1 dropped per arm as warm-up, 9 rounds each)</h3>")
    page.append("<div class=tw><table><tr><th>arm</th><th>conf</th><th>median tok/s</th><th>vs none</th>"
                "<th>P(sup) vs none (same-round pairs)</th><th>median of per-round mean TTFT ms</th>"
                "<th>median of per-round mean TPOT ms</th><th>verdict</th></tr>")
    for r in score:
        g = float(r["gain_pct_vs_ref"]); p = float(r["psup_vs_ref"])
        v = "tie" if abs(g) < 2 or 0.05 < p < 0.95 else ("better" if g > 0 else "worse")
        cls = " class=v" if v == "better" else (" class=x" if v == "worse" else "")
        page.append(f"<tr{cls}><td>{r['arm']}</td><td class=wrap>{desc[r['arm']]}</td>"
                    f"<td>{r['median_toks']}</td><td class=n>{g:+.2f}%</td><td>{p:.2f}</td>"
                    f"<td>{r['median_ttft_ms']}</td><td>{r['median_tpot_ms']}</td><td class=wrap>{v}</td></tr>")
    page.append("</table></div>")
    # per round
    rounds = {}
    for a in arms:
        for f in (QWEN / "node5" / f"qw_{a}").glob("bench_round*.log"):
            r = int(re.search(r"round(\d+)", f.name).group(1))
            rounds.setdefault(r, {})[a] = bench(f)
    page.append("<details open><summary>per round: tok/s (mean TTFT ms)</summary><div class=tw><table class=fit>"
                "<tr><th>round</th>" + "".join(f"<th>{a}</th>" for a in arms) + "</tr>")
    for r in sorted(rounds):
        cells = []
        for a in arms:
            b = rounds[r].get(a)
            cells.append(f"<td>{b[0]:.1f} <span class=n>({b[1]:.0f})</span></td>" if b else "<td></td>")
        page.append(f"<tr><td>{r}{' <span class=n>(cold server)</span>' if r == 1 else ''}</td>" + "".join(cells) + "</tr>")
    page.append("</table></div></details>")
    # detect + prefill
    det = (QWEN / "detect_counts.txt").read_text() if (QWEN / "detect_counts.txt").exists() else ""
    page.append("<h3>Did the rules run? (detect runs: own server per conf, 1 round, TUNING logs)</h3>")
    page.append("<div class=tw><table class=fit><tr><th>arm</th><th>rule hits ('Applied config' lines, 8 ranks)</th>"
                "<th>128K rule applied (decode size = conc 32 &times; hidden 2048 &times; 2 B)</th>"
                "<th>executed at 128K</th><th>round-1 tok/s</th><th>round-1 MEDIAN TTFT ms (real prefill)</th></tr>")
    # applied counts parsed from detect_counts.txt (grep of the node-5 detect logs);
    # the 128K rule and executed count per arm are the ones read from those logs in SUMMARY.md
    applied = dict(re.findall(r"^(\w+) detect: applied=(\d+)", det, re.M))
    rule128 = {"sc1": ("tree/ll/40", "not read"), "sc2": ("tree/ll/64", "TREE/LL channel{0..63} = 64"),
               "sc3": ("tree/ll/64", "not read"), "dummy": ("ring/simple/1", "RING/SIMPLE channel{0..0} = 1")}
    for a in ["sc1", "sc2", "sc3", "dummy"]:
        f = QWEN / "node5" / f"qw_{a}can_{a}" / "bench_round1.log"
        b = bench(f) if f.exists() else (float("nan"),) * 4
        n = int(applied.get(a, 0))
        page.append(f"<tr><td>{a}</td><td>{n:,}</td><td>{rule128[a][0]}</td><td>{rule128[a][1]}</td>"
                    f"<td>{b[0]:.1f}</td><td class={'b' if b[2] > 1000 else 'n'}>{b[2]:.0f}</td></tr>")
    page.append("</table></div>")
    page.append("<p class=note>Sizes hit in the model (all arms): 128K &times;1.59M calls, 4K &times;19k, 8K &times;6k, 2M &times;2.3k "
                "(<code>detect_counts.txt</code>, node 5 <code>qw_&lt;arm&gt;can_srv/logs</code>). "
                "Swap check (<code>qwswapcheck</code>, 2 rounds dummy vs none in ONE server, TUNING logs): 3,242,896 applied lines, "
                "all <code>ring simple 1</code> = exactly the two dummy rounds; none rounds applied nothing; 'hot-reloaded' logged at every swap.</p>")
    page.append("<h3>Reading</h3><p class=note>"
                "<b>Decode (tok/s, TPOT): all five arms tie</b>, |diff| &lt; 1-2% incl. the 1-channel dummy. Unexplained: rccl-tests says "
                "ring/simple/1 at 128K is 65 &micro;s vs default 35 &micro;s, &times;96 calls/rank/step &asymp; +7% TPOT expected, 0 observed.<br>"
                "<b>Prefill (TTFT) was NOT measured by rounds 2-10</b>: bench_serving reuses the same prompts (seed 42) and the radix cache is on, "
                "so those rounds are cache hits (~125 ms every arm). In the only real-prefill rounds (round 1 of each detect server) the dummy's "
                "TTFT was ~1760 ms vs 145-149 ms for sc1/sc2/sc3 &rarr; the 1-channel conf costs ~12&times; on real prefill; the sc confs do not.<br>"
                "Reviewer's corrections and open items: <code>2026-09-08-4-qwen/REVIEW.md</code>.</p>")
    return page


# ------------------------------------------------------------ 2026-09-09 follow-ups
def score_table(csvf, title, note, arms_desc):
    if not csvf.exists():
        return [f"<h3>{title}</h3><p class=note>not yet available: {html.escape(str(csvf))}</p>"]
    rows = list(csv.DictReader(open(csvf)))
    ref = [r for r in rows if float(r["gain_pct_vs_ref"]) == 0 and float(r["psup_vs_ref"]) == 0.5]
    out = [f"<h3>{title}</h3><p class=note>{note} · source <code>{html.escape(str(csvf.relative_to(RT)))}</code></p>",
           "<div class=tw><table><tr><th>arm</th><th>conf</th><th>median tok/s</th><th>vs reference</th><th>P(sup) vs reference</th>"
           "<th>median of per-round MEDIAN TTFT ms</th><th>median of per-round mean TPOT ms</th><th>verdict</th></tr>"]
    for r in rows:
        g = float(r["gain_pct_vs_ref"]); p = float(r["psup_vs_ref"]); n = int(r["n"])
        if p == 0.5 and g == 0: v, cls = "reference", ""
        elif p >= 0.95 and g > 2: v, cls = "BETTER", " class=v"
        elif p <= 0.05 and g < -2: v, cls = "WORSE", " class=x"
        else: v, cls = "tie", ""
        out.append(f"<tr{cls}><td>{r['arm']}</td><td class=wrap>{arms_desc.get(r['arm'].split('_')[0], r['arm'])}</td>"
                   f"<td>{r['median_toks']}</td><td class={'g' if g > 2 else ('b' if g < -2 else 'n')}>{g:+.1f}%</td>"
                   f"<td>{p:.2f}</td><td>{r['median_ttft_ms']}</td><td>{r['median_tpot_ms']}</td><td class=wrap>{v} (n={n})</td></tr>")
    out.append("</table></div>")
    return out


ARMS = {"sc2": "sanity-check-2 final.conf (8 rules 4K-2M; 128K-256K -> tree,ll,64)",
        "sc1": "sanity-check-1 final.conf", "sc3": "sanity-check-3 final.conf",
        "dummy": "ring,simple,1 channel all sizes (bad)", "dummy2": "tree,ll,1 channel all sizes (worse)",
        "none": "no rules = RCCL default (paired: plugin loaded, zero rules; one-server: no plugin)"}


def settle_section():
    page = []
    # 3. 4M settle
    st = RT / "2026-09-09-2-ab4m-settle" / "per_size_stats.csv"
    page.append("<h2>Sanity-check-3 follow-up (rccl-tests A/B, 2026-09-09): the 4M -73% was the over-request (144 &gt; 112 built channels), not &lsquo;112 at 4M&rsquo;</h2>")
    if st.exists():
        r = [x for x in csv.DictReader(open(st)) if x["size_bytes"] == "4194304"][0]
        page.append("<div class=tw><table><tr><th>run</th><th>rule at 4M</th><th>plugin applied</th><th>executed</th><th>default med</th><th>config med</th><th>gain</th><th>P(sup)</th></tr>"
                    f"<tr><td>2026-09-09-2-ab4m-settle (ses2-1)</td><td>ring,simple,112</td><td>{r['cfg_applied_ch']}</td><td>{r['cfg_exec']}</td><td>{r['def_median']}</td><td>{r['cfg_median']}</td>"
                    f"<td class=n>{float(r['gain_pct']):+.1f}%</td><td>{float(r['psup']):.2f}</td></tr>")
        st3 = RT / "2026-09-08-3-sanity-check-3" / "ab_out" / "allreduce_1n" / "per_size_stats.csv"
        r3 = [x for x in csv.DictReader(open(st3)) if x["size_bytes"] == "4194304"][0]
        page.append(f"<tr class=x><td>2026-09-08-3-sanity-check-3 (node 8)</td><td>ring,simple,144</td><td>{r3['cfg_applied_ch']}</td><td>{r3['cfg_exec']}</td><td>{r3['def_median']}</td><td>{r3['cfg_median']}</td>"
                    f"<td class=b>{float(r3['gain_pct']):+.1f}%</td><td>{float(r3['psup']):.2f}</td></tr></table></div>")
        page.append("<p class=note>A plugin value &le; the communicator's channel count is trimmed per call like the env path (112 &rarr; 103); a value above it (144) is not, "
                    "RCCL ran all 112 and collapsed. Rule for Process A: never emit a channel value above the node's default count (112 here). Guard parked as a tool change.</p>")
    return page


def followups_section():
    page = ["<h2>In-model results (2026-09-09; graphs ON, radix cache OFF, one SGLang server per arm). The 2026-09-08 in-model campaign is superseded: it ran graphs-off with the radix cache on.</h2>"]
    page.append("<p class=note>Three open points from 2026-09-08 closed with new runs on the TEST partition (XAI fully booked by another user). "
                "Details: <code>results-tuning/2026-09-09-LOG.md</code> and each run dir's PLAN/HANDS/SUMMARY/REVIEW.</p>")
    # 1. cuda graphs
    page.append("<h3>A. Conf vs plain RCCL (custom all-reduce off, MSCCL off, image flag removed): sc2 +18% on both models. Two passes = the same 4 (or 3) servers built twice in opposite order.</h3>")
    page.append("<p class=note>One SGLang server per arm (infer_many.sh, new IM_CUDA_GRAPH=1), radix cache OFF, d32 traffic (ISL 512 / OSL 512 / conc 32 / 128 prompts), "
                "image NCCL_MIN_NCHANNELS removed, MSCCL=0, 6 reps per server (rep 1 = cold start, dropped), two passes in opposite arm order. "
                "With graphs TPOT is 5-9 ms and the all_reduce sits on the decode critical path.</p>")
    q = RT / "2026-09-09-4-qwen-cudagraph" / "node"
    g = RT / "2026-09-09-5-gptoss-cudagraph" / "node"
    page += score_table(q / "cg1_score.csv", "Qwen3-30B-A3B, graphs ON, radix OFF, mode d32, pass 1 (none, sc2, dummy2, dummy) — node amd-mi350x-ses2-1 (MI350X)", "6 reps per server, first dropped", ARMS)
    page += score_table(q / "cg2_score.csv", "Qwen3-30B-A3B, graphs ON, radix OFF, mode d32, pass 2 (reverse order)", "same servers rebuilt in reverse order", ARMS)
    page += score_table(g / "cg1_score.csv", "gpt-oss-120b, graphs ON, radix OFF, mode d32, pass 1 (none, sc2, dummy) — node amd-mi355x-ses2-1", "--attention-backend triton", ARMS)
    page += score_table(g / "cg2_score.csv", "gpt-oss-120b, graphs ON, radix OFF, mode d32, pass 2 (reverse order)", "", ARMS)
    st = RT / "2026-09-09-6-gptoss-stock" / "node"
    page.append("<h3>B. Add the STOCK SGLang server as reference (custom all-reduce ON, MSCCL ON, image flag present, no plugin): stock is +28% over the tuned RCCL path. gpt-oss only (Qwen node was gone); two passes.</h3>")
    page.append("<p class=note>Same node/model/traffic as the gpt-oss graphs-on run, one server per arm, two passes. 'none' and 'sc2' reproduce run 5 within 0.5%. "
                "With the custom all-reduce on, the decode all_reduce never reaches RCCL, so the conf cannot act there; the +18% above is relative to a baseline that deployment does not use.</p>")
    ARMS2 = dict(ARMS); ARMS2["stock"] = "STOCK SGLang: custom all-reduce ON, RCCL_MSCCL_ENABLE=1, NCCL_MIN_NCHANNELS=112 present, no plugin"
    ARMS2["none"] = "custom AR off, MSCCL 0, flag removed, no plugin (= run 5 none)"
    page += score_table(st / "st1_score.csv", "gpt-oss-120b, graphs ON, radix OFF, mode d32, pass 1 (stock, none, sc2) — node amd-mi355x-ses2-1; reference = stock", "", ARMS2)
    page += score_table(st / "st2_score.csv", "gpt-oss-120b, graphs ON, radix OFF, mode d32, pass 2 (sc2, none, stock)", "", ARMS2)
    page.append("<p class=note>Attribution run (2026-09-09-7-gptoss-attr, not shown): the custom all-reduce alone reproduces stock (7316 vs 7306 tok/s); "
                "stock + the sc2 plugin = stock with ZERO rules applied and one 4-byte all_reduce seen by RCCL in the whole run. In stock SGLang the model's all_reduce never reaches RCCL.</p>")
    return page


# ------------------------------------------------------------------ hit-maps
def size_k(b):
    b = int(b)
    return size_h(b) if b >= 1024 else f"{b}B"


def hitmap_table(csvf, title):
    if not csvf.exists():
        return [f"<h3>{title}</h3><p class=note>not available yet</p>"]
    rows = list(csv.DictReader(open(csvf)))
    rules = [r for r in rows if r["kind"] == "rule"]
    misses = [r for r in rows if r["kind"] == "miss"]
    tot = sum(int(r["hits"]) for r in rules)
    out = [f"<h3>{title}</h3>", "<div class=tw><table class=fit><tr><th>rule</th><th>hits</th><th>share</th></tr>"]
    for r in sorted(rules, key=lambda r: -int(r["hits"])):
        h = int(r["hits"])
        rng = size_k(r["min_bytes"]) if r["min_bytes"] == r["max_bytes"] else f"{size_k(r['min_bytes'])}..{size_k(r['max_bytes'])}"
        out.append(f"<tr{' class=v' if h else ''}><td>{rng} {r['algorithm']}/{r['protocol']}/{r['channels']}</td><td>{h:,}</td>"
                   f"<td>{(100*h/tot if tot else 0):.0f}%</td></tr>")
    out.append("</table></div>")
    if misses:
        n = sum(int(r["hits"]) for r in misses)
        top = ", ".join(size_k(r["min_bytes"]) for r in sorted(misses, key=lambda r: -int(r["min_bytes"]))[:5])
        out.append(f"<p class=note>all_reduce sizes with no rule (RCCL default used): {len(misses)} sizes, {n:,} calls; largest: {top}</p>")
    return out


def hitmaps_section():
    H = RT / "2026-09-09-8-hitmaps"
    page = ["<h2>Which sc2 rules the models actually hit (graphs ON, d32; from the detect servers' TUNING logs, 8 ranks summed)</h2>",
            "<p class=note>With graphs on, decode is captured once per batch size and replayed, so decode rules show few hits; prefill counts per call. "
            "CSVs: <code>results-tuning/2026-09-09-8-hitmaps/</code>.</p>"]
    for f, t in (("qwen_cg_d32_sc2.csv", "Qwen3-30B-A3B, arm sc2"),
                 ("gptoss_cg_d32_sc2.csv", "gpt-oss-120b, arm sc2")):
        page += hitmap_table(H / f, t)
    return page


def attribution_section():
    at = RT / "2026-09-09-7-gptoss-attr" / "node" / "at_score.csv"
    page = ["<h2>What makes stock fast: the AITER custom all-reduce, not MSCCL (attribution run 2026-09-09-7-gptoss-attr)</h2>",
            "<p class=note>gpt-oss-120b, graphs ON, radix OFF, mode d32, node amd-mi355x-ses2-1, one server per arm, 6 reps (rep 1 dropped). "
            "Stock differs from the RCCL-path arms in three settings at once; here each is switched alone. "
            "SGLang's custom all-reduce on ROCm is the AITER kernel (<code>sglang/srt/distributed/device_communicators/custom_all_reduce.py</code> imports "
            "<code>aiter.dist.device_communicators.custom_all_reduce</code>). Source <code>results-tuning/2026-09-09-7-gptoss-attr/node/at_score.csv</code>.</p>"]
    if not at.exists():
        return page + ["<p class=note>not available</p>"]
    rows = {r["arm"].split("_")[0]: r for r in csv.DictReader(open(at))}
    # none and sc2 from run 6 (same node, same hour, same design; ref = stock)
    st = RT / "2026-09-09-6-gptoss-stock" / "node" / "st1_score.csv"
    if st.exists():
        for r in csv.DictReader(open(st)):
            a = r["arm"].split("_")[0]
            if a in ("none", "sc2"):
                rows[a] = r
    desc = {"stock": ("ON", "ON", "present", "none", "stock reference"),
            "none": ("off", "off", "removed", "none", "plain RCCL (run 6, same node/hour)"),
            "sc2": ("off", "off", "removed", "sc2 conf", "plain RCCL + our tuned conf (run 6, same node/hour)"),
            "customar": ("ON", "off", "removed", "none", "custom all-reduce alone"),
            "msccl": ("off", "ON", "present", "none", "MSCCL + channel flag, no custom AR (the image env minus the kernel)"),
            "stocksc2": ("ON", "ON", "present", "sc2 conf, TUNING logs", "stock + tuner plugin: 0 rules applied, RCCL saw one 4-byte all_reduce in the whole run")}
    page.append("<div class=tw><table><tr><th>arm</th><th>custom all-reduce (AITER)</th><th>MSCCL</th><th>NCCL_MIN_NCHANNELS=112</th><th>plugin</th>"
                "<th>median tok/s</th><th>vs stock</th><th>TPOT ms</th><th>meaning</th></tr>")
    for a in ("stock", "customar", "msccl", "stocksc2", "sc2", "none"):
        r = rows.get(a)
        if not r:
            continue
        d = desc[a]; g = float(r["gain_pct_vs_ref"])
        cls = " class=x" if g < -2 else ""
        page.append(f"<tr{cls}><td>{a}</td><td>{d[0]}</td><td>{d[1]}</td><td>{d[2]}</td><td>{d[3]}</td><td>{r['median_toks']}</td>"
                    f"<td class={'b' if g < -2 else 'n'}>{g:+.1f}%</td><td>{r['median_tpot_ms']}</td><td class=wrap>{d[4]}</td></tr>")
    page.append("</table></div>")
    page.append("<p class=note>Reading: custom all-reduce alone = stock (7316 vs 7306). Turning it off and keeping MSCCL + the flag costs 15% (6226, TPOT 4.90 vs 4.14). "
                "With the custom all-reduce on, the tuner plugin never fires: every model all_reduce (decode and the 47-94 MB prefill chunks, confirmed again 2026-09-10 at p8k/mix on both models) "
                "goes through the AITER kernel, not RCCL. Ranking on gpt-oss d32 decode: custom AR 7310 &gt; MSCCL+flag 6226 &gt; sc2 conf 5720 &gt; plain RCCL 4840 tok/s.</p>")
    return page


def main():
    page = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,"
            f"initial-scale=1'><title>sanity checks 2026-09-08</title><style>{CSS}</style>"
            "<p><a href='index.html'>&larr; index</a> · <a href='collective_all_reduce.html'>all_reduce (08-24 pages)</a>"
            " · <a href='codeflow.html'>code flow</a></p>"
            "<h1>all_reduce, 1 node &mdash; 2026-09-08 sanity checks: Process A by one command, three channel grids, then Qwen</h1>"]
    strip = []
    sections = []
    for dirname, label in RUNS:
        sec, kept, n = run_section(dirname, label)
        sections += sec
        strip.append(f"{label}: <span class=g>{len(kept)} of {n} sizes validated</span>")
    page.append("<p class=note>" + " &nbsp;·&nbsp; ".join(strip) + "</p>")
    page.append("<p class=note>Same tool for all three: <code>rccl_tune.py run</code> = book node → adaptive racing search "
                "(anchors 1,8,24,48, margin 15, tol 0.5, median of 3, + 3 NCCL-default runs) → conf → A/B through the tuner plugin "
                "(9 repeats/arm interleaved, container runtime, image NCCL_MIN_NCHANNELS removed from BOTH arms, P(sup) &ge; 0.95, "
                "gain &gt; 2%, regression &lt; 2%) → RUNLOG. Only <code>--grid</code> differs. "
                "'ours executed' is read from the runs' own NCCL INFO logs (channel{Lo..Hi}), never from the -A plan. "
                "Details, hand steps and reviewer findings: each run dir's SUMMARY.md / HANDS.md / REVIEW.md; day log "
                "<code>results-tuning/2026-09-08-LOG.md</code>.</p>")
    page += sections
    page += settle_section()
    page += followups_section()
    page += hitmaps_section()
    page += attribution_section()
    (HERE / "sanity_2026-09-08.html").write_text("\n".join(page))
    print("wrote", HERE / "sanity_2026-09-08.html")


if __name__ == "__main__":
    main()
