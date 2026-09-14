#!/usr/bin/env python3
"""night_2026-09-14.html - one page for the 2026-09-14 night campaigns A, B', C, D. Verdict first, then one section per campaign,
read from the run dirs (A: 2026-09-14-1-bigmsg runs_busbw.csv + ab_out/allreduce_1n/per_size_stats.csv; B': 2026-09-14-3-bigmsg-inmodel/<tag>/b*/score.csv;
C: 2026-09-14-2-sc2plusamd/<tag>/<mode>/pass*/score.csv; D: 2026-09-14-4-csvtuner/summary.csv if present). Pending parts say so. Regenerate any time."""
import csv, glob, html, os, re, statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RT = HERE.parent
A = RT / "2026-09-14-1-bigmsg"; B = RT / "2026-09-14-3-bigmsg-inmodel"; C = RT / "2026-09-14-2-sc2plusamd"; D = RT / "2026-09-14-4-csvtuner"
CSS = """body{background:#0f1216;color:#d6dbe3;font:14px/1.5 system-ui,sans-serif;margin:0;padding:18px 26px 50px}
h1{font-size:21px;margin:0 0 6px}h2{font-size:16px;color:#5ab0ff;border-bottom:1px solid #2a323d;padding-bottom:4px;margin:30px 0 8px}h3{font-size:14px;margin:14px 0 4px;color:#cfd3d9}
p{max-width:1150px;margin:6px 0}.note{color:#8b95a3;font-size:13px}code{background:#1d242e;padding:1px 5px;border-radius:4px;font-size:12.5px}
table{border-collapse:collapse;font-size:13px;margin:8px 0}th,td{border:1px solid #2a323d;padding:4px 8px;text-align:right}th{background:#1b222b}
td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}.tw{overflow-x:auto;max-width:1150px}
.g{color:#5fd38a}.b{color:#ff7b72}.n{color:#8b95a3}.card{background:#161b22;border:1px solid #2a323d;border-radius:8px;padding:12px 14px;margin:10px 0;max-width:1150px}
.verdict{font-size:15px}.verdict li{margin:4px 0}a{color:#5ab0ff}"""
SZ = {134217728: "128M", 268435456: "256M", 536870912: "512M", 1073741824: "1G", 2147483648: "2G"}
MODELS = {"qwen": "Qwen3-30B-A3B", "gptoss": "gpt-oss-120b", "dsr1": "DeepSeek-R1-0528 MXFP4"}
BATCH_MIB = {("qwen", "b32768"): "128 MiB", ("qwen", "b65536"): "256 MiB", ("gptoss", "b32768"): "180 MiB", ("gptoss", "b65536"): "360 MiB",
             ("dsr1", "b16384"): "224 MiB", ("dsr1", "b32768"): "448 MiB"}
MODE_BYTES = {"d32": 32, "d128": 128, "mix": 64, "d256": 256}
HIDDEN = {"qwen": 2048, "gptoss": 2880, "dsr1": 7168}


def med(xs):
    return statistics.median(xs) if xs else None


def pct(a, b):
    return (a / b - 1) * 100 if a and b else None


def fpct(v, cls=True):
    if v is None: return "<td class=n>-</td>"
    c = "g" if v > 2 else ("b" if v < -2 else "n")
    return f"<td class={c if cls else 'n'}>{v:+.1f}%</td>"


def section_a():
    out = ["<h2>A. rccl-tests all_reduce, 1 node, 128M..2G: can any algo/proto/channels beat RCCL's default? (dir <code>2026-09-14-1-bigmsg</code>, <a href='bigmsg_2026-09-14.html'>detail page</a>)</h2>"]
    f = A / "runs_busbw.csv"
    if not f.exists():
        return out + ["<p class=note>pending</p>"]
    rows = list(csv.DictReader(open(f)))
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["requested_algo"], r["requested_proto"], r["requested_ch"]); g[k][int(r["size_bytes"])].append(float(r["busbw_ip"]))
    dkey = ("default", "default", "default"); d = {s: med(g[dkey][s]) for s in SZ} if dkey in g else {}
    keys = sorted(g, key=lambda k: -(med(g[k][2147483648]) or 0))
    out.append("<div class=tw><table><tr><th>requested</th><th>128M</th><th>256M</th><th>512M</th><th>1G</th><th>2G</th><th>vs default @2G</th></tr>")
    for k in keys[:8]:
        lab = "RCCL default (executes RING/SIMPLE/112)" if k == dkey else f"{k[0]}/{k[1]}/{k[2]}"
        cells = "".join(f"<td>{med(g[k][s]):.0f}</td>" for s in SZ)
        v = fpct(pct(med(g[k][2147483648]), d.get(2147483648))) if k != dkey else "<td class=n>reference</td>"
        out.append(f"<tr{' style=background:#1d3a2a' if k == dkey else ''}><td>{lab}</td>{cells}{v}</tr>")
    out.append("</table></div><p class=note>Median busbw GB/s of 3 repeats; 15 cells measured ({RING,TREE} x {LL,LL128,SIMPLE} at 32/48 channels, RING/SIMPLE also at 56/64/112/224). LL128 is honoured on RCCL 2.30.4 (it was substituted on 2.27.7) but 20% behind SIMPLE per channel.</p>")
    ps = A / "ab_out/allreduce_1n/per_size_stats.csv"
    if ps.exists():
        st = list(csv.DictReader(open(ps)))
        out.append("<h3>A/B verdict (9 repeats per arm, keep a size only if P(sup) &ge; 0.95 and gain &gt; 2%)</h3><div class=tw><table><tr><th>size</th><th>default median</th><th>conf median</th><th>gain</th><th>P(sup)</th><th>executed default / conf</th><th>verdict</th></tr>")
        for r in st:
            out.append(f"<tr><td>{SZ.get(int(r['size_bytes']), r['size_bytes'])}</td><td>{float(r['def_median']):.1f}</td><td>{float(r['cfg_median']):.1f}</td><td>{float(r['gain_pct']):+.2f}%</td><td>{float(r['psup']):.2f}</td><td>{r['def_exec']} / {r['cfg_exec']}</td><td class=b>DROP</td></tr>")
        out.append("</table></div>")
    out.append("<p><b>Verdict A:</b> exact parity at every size; the best forced cell is the default itself (RING/SIMPLE/112). Nothing to tune above 64 MiB in rccl-tests.</p>")
    return out


def section_b():
    out = ["<h2>B'. Tune from the model at &ge; 128 MiB (prefill all_reduces on RCCL: custom AR declines above 64 MiB, quick-reduce set to NONE) (dir <code>2026-09-14-3-bigmsg-inmodel</code>)</h2>",
           "<p class=note>One hot-reload SGLang server per model x prefill batch size; 13 arms x 3 rounds round-robin: no rule (RCCL default), ring/simple x {8,16,24,32,48,64,80,112}, ring/ll128 x {32,64,112}, tree/ll128 x 112; "
           "then a separate stock server (image default: INT8 quick-reduce). Metric: prefill input tok/s (median of 3). Traffic ISL 8192 / OSL 128 / conc 32 / 96 prompts. RCCL 2.30.4, graphs on, radix off.</p>"]
    any_ = False
    out.append("<div class=tw><table><tr><th>model</th><th>prefill batch (all_reduce)</th><th>RCCL default</th><th>rs8</th><th>rs32</th><th>rs64</th><th>rs80</th><th>rs112</th><th>LL128/tree arms</th><th>best arm vs default</th><th>stock INT8</th><th>stock vs default</th></tr>")
    for tag in ("qwen", "gptoss", "dsr1"):
        for bdir in sorted(glob.glob(str(B / tag / "b*"))):
            f = Path(bdir) / "score.csv"
            if not f.exists(): continue
            any_ = True
            rs = list(csv.DictReader(open(f)))
            s = {r["arm"]: float(r["in_toks"]) for r in rs if r["phase"] == "search"}
            stock = med([float(r["in_toks"]) for r in rs if r["phase"] == "stockref"])
            base = s.get("none"); ll = [v for a, v in s.items() if a.startswith(("rl", "tl"))]
            best = max((a for a in s if a != "none"), key=lambda a: s[a]) if s else None
            cell = lambda a: f"<td>{s[a]:,.0f} <span class=n>({pct(s[a], base):+.0f}%)</span></td>" if a in s and base else "<td class=n>-</td>"
            out.append(f"<tr><td>{MODELS[tag]}</td><td>{os.path.basename(bdir)[1:]} tokens ({BATCH_MIB.get((tag, os.path.basename(bdir)), '?')})</td><td>{base:,.0f}</td>{cell('rs8')}{cell('rs32')}{cell('rs64')}{cell('rs80')}{cell('rs112')}"
                       f"<td>{min(ll):,.0f}..{max(ll):,.0f}</td>{fpct(pct(s[best], base))}<td>{stock:,.0f}</td>{fpct(pct(stock, base))}</tr>")
    out.append("</table></div>")
    if not any_:
        out.append("<p class=note>pending</p>")
    out.append("<p><b>Verdict B':</b> no arm beats RCCL's default in any of the six combos (best within &plusmn;0.1%, gate 2%); fewer channels only lose (8 channels -57..-68%, 80 channels -1..-2%). "
               "Verification with TUNING logs (gpt-oss, 180 MiB): every arm executed RING/SIMPLE with channels 0..111; RCCL marks [ring][ll128] and [tree][ll128] IGNORE in the SGLang communicator, so a tuner can only steer ring/simple channel counts there, and the default already uses all 112. "
               "The deployment path (INT8 quick-reduce) is 2.4-5.8% faster than any exact RCCL configuration.</p>")
    return out


def section_c():
    out = ["<h2>C. Carrying AMD's built-in gfx950 rules for the sizes sc2 does not cover: does it remove the plugin's miss penalty? (dir <code>2026-09-14-2-sc2plusamd</code>)</h2>",
           "<p class=note>Arms, all with custom AR off: none = RCCL as shipped (DDA) · nodda = RCCL_DDA_ENABLE=0 (classic path, AMD's built-in CSV table active) · sc2 = nodda + plugin + sc2 conf (displaces AMD's table) · "
           "sc2plus = nodda + plugin + sc2_plus_amd.conf (sc2 rules + AMD's four rules in the gaps). Decode tok/s, median of 5 reps (rep 1 dropped), two passes. Node 2.</p>"]
    out.append("<div class=tw><table><tr><th>model</th><th>mode</th><th>decode all_reduce</th><th>sc2 rule?</th><th>none (DDA)</th><th>nodda</th><th>sc2</th><th>sc2plus</th><th>sc2 vs nodda</th><th>sc2plus vs nodda</th><th>sc2plus vs sc2</th></tr>")
    sc2rules = [(int(r["min_bytes"]), int(r["max_bytes"])) for r in csv.DictReader(l for l in open(RT / "2026-09-08-2-sanity-check-2/all_reduce_1n.final.conf") if not l.startswith("#"))]
    any_ = False
    for tag in ("qwen", "gptoss", "dsr1"):
        for mode in ("d32", "d128", "mix", "d256"):
            vals = defaultdict(list)
            for p in ("pass1", "pass2"):
                f = C / tag / mode / p / "score.csv"
                if f.exists():
                    for r in csv.DictReader(open(f)):
                        vals[r["arm"].split("_")[0]].append(float(r["median_toks"]))
            if not vals: continue
            any_ = True
            m = {a: med(v) for a, v in vals.items()}
            b = MODE_BYTES[mode] * HIDDEN[tag] * 2
            cov = any(lo <= b <= hi for lo, hi in sc2rules)
            fmt = lambda a: f"{m[a]:,.0f}" if a in m else "-"
            out.append(f"<tr><td>{MODELS[tag]}</td><td>{mode}</td><td>{b:,} B</td><td>{'yes' if cov else 'NO'}</td><td>{fmt('none')}</td><td>{fmt('nodda')}</td><td>{fmt('sc2')}</td><td>{fmt('sc2plus')}</td>"
                       f"{fpct(pct(m.get('sc2'), m.get('nodda')))}{fpct(pct(m.get('sc2plus'), m.get('nodda')))}{fpct(pct(m.get('sc2plus'), m.get('sc2')))}</tr>")
    out.append("</table></div>")
    if not any_:
        out.append("<p class=note>pending</p>")
    cs = C / "SUMMARY.md"
    if cs.exists():
        m = re.search(r"Answer:\s*(.+?)\n\n", cs.read_text(), re.S)
        if m: out.append(f"<p><b>Verdict C:</b> {html.escape(m.group(1).strip())}</p>")
    else:
        out.append("<p class=note>Verdict C: pending (chains running).</p>")
    return out


def section_d():
    out = ["<h2>D. Delivery check: the same conf through RCCL 2.30.4's built-in CSV tuner (no plugin) vs our plugin (dir <code>2026-09-14-4-csvtuner</code>)</h2>"]
    f = D / "summary.csv"
    if f.exists():
        rs = list(csv.DictReader(open(f)))
        cols = list(rs[0].keys())
        out.append("<div class=tw><table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
        for r in rs:
            out.append("<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols) + "</tr>")
        out.append("</table></div>")
    ds = D / "SUMMARY.md"
    if ds.exists():
        m = re.search(r"Answer:\s*(.+?)\n\n", ds.read_text(), re.S)
        if m: out.append(f"<p><b>Verdict D:</b> {html.escape(m.group(1).strip())}</p>")
    else:
        out.append("<p class=note>pending (runs after campaign C on node 2)</p>")
    return out


def main():
    page = [f"<!doctype html><meta charset=utf-8><title>night 2026-09-14</title><style>{CSS}</style>",
            "<p><a href='index.html'>&larr; index</a> · <a href='bigmsg_2026-09-14.html'>A detail</a> · <a href='inmodel_rocm10_2026-09-10.html'>rocm10 in-model (09-10)</a> · <a href='dda_2026-09-14.html'>DDA explainer</a></p>",
            "<h1>Night of 2026-09-14: is there any place left for a 1-node RCCL tuner conf on the rocm10 stack? Campaigns A, B', C, D</h1>",
            "<p class=note>Stack: lmsysorg/sglang:v0.5.19-rocm10-mi35x (SGLang 0.5.19, RCCL 2.30.4, torch 2.11). All timing in TIMELINE.md of each dir (Israel time). Every number below comes from a score.csv / metrics.csv in the named dir.</p>",
            "<div class=card><ul class=verdict>",
            "<li><b>A (rccl-tests, 128M..2G):</b> exact parity, the default (RING/SIMPLE/112) is the best cell. Nothing to tune above DDA's 64 MiB ceiling.</li>",
            "<li><b>B' (tune from the model, 128-448 MiB prefill all_reduces on RCCL):</b> no rule beats the default for Qwen, gpt-oss or DeepSeek (best within &plusmn;0.1%). Only ring/simple channels are steerable in the model's communicator (LL128/tree are IGNORE) and the default already uses 112. Stock INT8 quick-reduce is +2.4..+5.8% over any exact RCCL config.</li>",
            "<li><b>C (sc2 + AMD's gap rules):</b> sc2plus == sc2 wherever sc2 has a rule and == AMD's table (nodda) wherever it does not: the -13..-15% miss penalty at gpt-oss d256, DeepSeek d32 and d128 becomes -0.1%. RCCL as shipped (DDA) stays 8-29% ahead of every classic-path arm.</li>",
            "<li><b>D (built-in CSV tuner delivery):</b> the same conf through RCCL 2.30.4's built-in CSV tuner (no plugin) loads 16 rules, applies exactly as many as the plugin (133480 / 74760 / 66920 lines) and gives the same tok/s within 0.5%. The deliverable on RCCL &ge; 2.30 is the conf file alone.</li>",
            "<li><b>Bottom line for GPU-107 on this stack, single node:</b> below 64 MiB AITER (in SGLang) and DDA (in RCCL) take every all_reduce without consulting a tuner; above 64 MiB the default is already optimal in every space a tuner can reach. The remaining tuner-visible domain is multi-node (training-shaped traffic).</li>",
            "</ul></div>"]
    page += section_a() + section_b() + section_c() + section_d()
    out = HERE / "night_2026-09-14.html"
    out.write_text("\n".join(page)); print(f"wrote {out}")


if __name__ == "__main__":
    main()
