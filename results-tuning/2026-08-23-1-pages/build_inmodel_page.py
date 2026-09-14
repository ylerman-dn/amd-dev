#!/usr/bin/env python3
"""inmodel_2026-09-10.html — the in-model campaign page agreed 2026-09-09 (AGREED.md in the run dir):
1) the sc2 conf and how it was derived, 2) per model -> per mode -> pass 1 / pass 2 tables, 3) hit-map per model-mode for sc2 and dummy.
Everything is read from results-tuning/2026-09-10-1-inmodel/<model>/<mode>/{pass1,pass2}/score.csv and detect/hitmap_*.csv.
Reuses the CSS/table helpers of build_sanity_pages.py (importing it does not build anything)."""
import csv, html, os, sys
from pathlib import Path
from build_sanity_pages import CSS, size_h, size_k, hitmap_table

HERE = Path(__file__).parent
RT = HERE.parent
# argv: [run_dir_name] [output_html] [stack description]   (defaults = the 2026-09-10 campaign on the v0.5.17/rocm720 image)
RUN = RT / (sys.argv[1] if len(sys.argv) > 1 else "2026-09-10-1-inmodel")
OUT = sys.argv[2] if len(sys.argv) > 2 else "inmodel_2026-09-10.html"
STACK = sys.argv[3] if len(sys.argv) > 3 else "image lmsysorg/sglang:v0.5.17-rocm720-mi35x (SGLang 0.5.17, ROCm 7.2.0, RCCL 2.27.7 @0d2c4fd 2025-12-09)"
SC2 = RT / "2026-09-08-2-sanity-check-2"
MODELS = [("qwen", "Qwen3-30B-A3B (TP-8, hidden 2048: decode all_reduce = conc x 4096 B)"),
          ("gptoss", "gpt-oss-120b (TP-8, hidden 2880: decode all_reduce = conc x 5760 B; --attention-backend triton)")]
HIDDEN = {"qwen": 2048, "gptoss": 2880}
MODES = [("d32", "decode-heavy: ISL 512 / OSL 512 / concurrency 32 / 128 prompts"),
         ("p8k", "prefill-heavy: ISL 8192 / OSL 128 / concurrency 32 / 96 prompts"),
         ("d128", "decode-heavy, big batch: ISL 512 / OSL 512 / concurrency 128 / 384 prompts"),
         ("mix", "mixed: ISL 2048 / OSL 512 / concurrency 64 / 192 prompts")]
CONC = {"d32": 32, "p8k": 32, "d128": 128, "mix": 64}
# a run dir may override: models.txt "tag|description|hidden", modes.txt "mode|description|concurrency"
if (RUN / "models.txt").exists():
    MODELS, HIDDEN = [], {}
    for l in (RUN / "models.txt").read_text().splitlines():
        if l.strip():
            t, d, h = l.split("|"); MODELS.append((t, d)); HIDDEN[t] = int(h)
if (RUN / "modes.txt").exists():
    MODES, CONC = [], {}
    for l in (RUN / "modes.txt").read_text().splitlines():
        if l.strip():
            m, d, c = l.split("|"); MODES.append((m, d)); CONC[m] = int(c)
ROCM10 = (RUN / "models.txt").exists()   # the 2026-09-10-2 rerun on RCCL 2.30.4 (5 arms); the old 4-arm page keeps its texts
ARMS = {"stock": "STOCK SGLang: AITER custom all-reduce ON, image env untouched (NCCL_MIN_NCHANNELS=112), no plugin",
        "none": ("RCCL as shipped: custom all-reduce off, NCCL_MIN_NCHANNELS removed, no plugin"
                 + (" (on RCCL 2.30.4 its DDA path takes the decode all_reduces and never consults a tuner)" if ROCM10 else "")),
        "nodda": "RCCL with DDA off (RCCL_DDA_ENABLE=0): the classic ring/tree path, the only one a tuner plugin can steer; no plugin",
        "sc2": ("RCCL (DDA off)" if ROCM10 else "plain RCCL") + " + tuner plugin + sc2 conf",
        "dummy": ("RCCL (DDA off)" if ROCM10 else "plain RCCL") + " + tuner plugin + dummy conf (ring,simple,1 channel for all sizes)"}
ARM_ORDER = ("stock", "none", "nodda", "sc2", "dummy") if ROCM10 else ("stock", "none", "sc2", "dummy")
ORDER_NOTE_ROCM10 = ("<p class=note>Order in every table: stock &gt; none (RCCL as shipped, DDA path) &gt; sc2 &ge; nodda &gt;&gt; dummy. "
                     "The conf still lifts the classic RCCL path where a rule covers the decode size, but RCCL 2.30.4's own DDA all-reduce beats the tuned classic path everywhere below 2 MB, "
                     "and stock SGLang (AITER custom all-reduce) stays ahead of both.</p>")
BASE = "nodda" if ROCM10 else "none"   # the arm the plugin arms are built on = the tuner's own effect is sc2 vs BASE
def gap_label(b):
    if 262144 < b < 524288: return "NO RULE (gap 256K..512K)"
    if b > 1048576 and b != 2097152: return "NO RULE (above 1M only the exact 2M rule exists)"
    return "NO RULE"
ORDER = {"pass1": ", ".join(ARM_ORDER), "pass2": ", ".join(reversed(ARM_ORDER))}
NODE = {("qwen", "d32"): "amd-mi355x-8", ("qwen", "p8k"): "amd-mi355x-8", ("qwen", "d128"): "amd-mi355x-9", ("qwen", "mix"): "amd-mi355x-9",
        ("gptoss", "d32"): "amd-mi355x-des2-2", ("gptoss", "p8k"): "amd-mi355x-des2-2", ("gptoss", "d128"): "amd-mi355x-ses2-1", ("gptoss", "mix"): "amd-mi355x-ses2-1"}
if (RUN / "nodes.txt").exists():   # "<model> <mode> <node>" per line overrides the map above
    NODE = {(a, b): c for a, b, c in (l.split() for l in (RUN / "nodes.txt").read_text().splitlines() if l.strip())}


def pass_table(csvf, title):
    if not csvf.exists():
        return [f"<h4>{title}</h4><p class=note>not available yet</p>"]
    rows = list(csv.DictReader(open(csvf)))
    out = [f"<h4>{title}</h4>",
           "<div class=tw><table><tr><th>arm</th><th>server</th><th>median tok/s</th><th>vs stock</th><th>P(sup) vs stock</th>"
           "<th>TTFT ms (median of per-rep mean)</th><th>TPOT ms</th><th>verdict</th></tr>"]
    for want in ARM_ORDER:
        r = next((x for x in rows if x["arm"].split("_")[0] == want), None)
        if not r:
            out.append(f"<tr><td>{want}</td><td class=wrap>{ARMS[want]}</td><td colspan=6 class=n>missing</td></tr>")
            continue
        g = float(r["gain_pct_vs_ref"]); p = float(r["psup_vs_ref"]); n = int(r["n"])
        if want == "stock": v, cls = "reference", ""
        elif p >= 0.95 and g > 2: v, cls = "BETTER than stock", " class=v"
        elif p <= 0.05 and g < -2: v, cls = "worse than stock", " class=x"
        else: v, cls = "tie with stock", ""
        out.append(f"<tr{cls}><td>{want}</td><td class=wrap>{ARMS[want]}</td><td>{r['median_toks']}</td>"
                   f"<td class={'g' if g > 2 else ('b' if g < -2 else 'n')}>{g:+.1f}%</td><td>{p:.2f}</td>"
                   f"<td>{r['median_ttft_ms']}</td><td>{r['median_tpot_ms']}</td><td class=wrap>{v} (n={n})</td></tr>")
    out.append("</table></div>")
    # sc2 vs none, the tuner question, as one line
    sc2 = next((x for x in rows if x["arm"].startswith("sc2")), None)
    base = next((x for x in rows if x["arm"].startswith(BASE)), None)
    none = next((x for x in rows if x["arm"].startswith("none")), None)
    if sc2 and base and float(base["median_toks"]):
        d = (float(sc2["median_toks"]) / float(base["median_toks"]) - 1) * 100
        txt = f"sc2 vs {BASE} (the tuner's own effect, same RCCL path): {d:+.1f}% tok/s, TPOT {sc2['median_tpot_ms']} vs {base['median_tpot_ms']} ms."
        if ROCM10 and none and float(sc2["median_toks"]):
            txt += f" none (RCCL as shipped, DDA) vs sc2: {(float(none['median_toks']) / float(sc2['median_toks']) - 1) * 100:+.1f}% tok/s."
        out.append(f"<p class=note>{txt}</p>")
    return out


def conf_section():
    conf = (SC2 / "all_reduce_1n.final.conf").read_text().strip()
    page = ["<h2>1. The conf under test: sc2 (sanity-check-2 final config) and how it was made</h2>",
            f"<pre class=keep>{html.escape(conf)}</pre>",
            "<p class=note>Format: collective, min_bytes, max_bytes, algorithm, protocol, channels, nNodes, nRanks, numPipeOps, regBuff. "
            "Every rule says: for an all_reduce of this size on 1 node / 8 ranks, use this algo/proto and at most this many channels.</p>",
            "<h3>How it was derived (2026-09-08-2-sanity-check-2, one command: <code>rccl_tune.py run --collectives all_reduce --scales 1 --grid 1..112</code>)</h3>",
            "<ol class=note>"
            "<li><b>Search</b> (rccl-tests <code>all_reduce_perf</code>, stock RCCL inside the SGLang image, one MI355X node, 8 GPUs): racing search over "
            "{RING,TREE} x {LL,SIMPLE} x channels 1..112 (18 values), 18 message sizes 4K..512M, 3 repeats per point, 120 runs (26% fewer than the full grid), "
            "plus 3 runs of RCCL default. Winner per size = highest busbw within 0.5%, fewest channels among ties.</li>"
            "<li><b>Conf</b>: one rule per size from the winners (18 rules).</li>"
            "<li><b>A/B through the tuner plugin</b>: each rule vs RCCL default, 9 repeats per arm interleaved, both arms with the image's NCCL_MIN_NCHANNELS removed. "
            "Kept only rules with P(sup) &ge; 0.95, gain &gt; 2%, no size regressing &gt; 2%: 10 of 18 sizes (4K..2M, +12..+74% busbw); the 8 sizes &ge; 4M were dropped (8M..512M parity, 4M a real -1.1% loss).</li>"
            "<li><b>Final</b>: adjacent identical rules merged &rarr; the 8 rules above. Full detail: <a href='sanity_2026-09-08.html'>sanity_2026-09-08.html</a>.</li></ol>",
            "<p class=note>Note the channel numbers are requests: RCCL trims per call (e.g. 48 requested at 8K runs 4). What matters is the executed behaviour, "
            "which the A/B measured (column 'ours executed' on the sanity page).</p>"]
    return page


def main():
    page = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>in-model {RUN.name}</title><style>{CSS} h4{{font-size:13px;color:#cfd3d9;margin:14px 0 4px}}</style>"
            "<p><a href='index.html'>&larr; index</a> · <a href='sanity_2026-09-08.html'>sanity checks + first in-model results</a></p>"
            + (f"<h1>Tuner conf in live serving on RCCL 2.30.4 &mdash; {len(MODELS)} models, {len(MODES)} traffic modes, {len(ARM_ORDER)} arms ({html.escape(RUN.name)})</h1>" if ROCM10 else
             f"<h1>Tuner conf in live serving &mdash; Qwen3-30B-A3B and gpt-oss-120b, 4 traffic modes, 4 arms ({html.escape(RUN.name)})</h1>"),
            f"<p class=note><b>Stack:</b> {html.escape(STACK)}</p>",
            "<p class=note>Measured the deployment way: SGLang with CUDA graphs ON, radix cache OFF, one server per arm, 6 benchmark reps per server "
            "(rep 1 = cold start, dropped), every model/mode run twice with the arm order reversed (pass 2) as a drift check. "
            f"Reference arm = stock SGLang. Data: <code>results-tuning/{RUN.name}/&lt;model&gt;/&lt;mode&gt;/{{pass1,pass2}}/score.csv</code>; "
            "timeline in <code>TIMELINE.md</code> there.</p>"]
    # overview table: medians of both passes, sc2 vs none, stock vs sc2
    hidden = HIDDEN
    rules = [r for r in csv.DictReader(l for l in open(SC2 / "all_reduce_1n.final.conf") if not l.startswith("#"))]
    page.append("<h2>0. Overview: median tok/s per arm, pass 1 / pass 2 (every table below in one line each)</h2>")
    if ROCM10:
        page.append("<div class=tw><table><tr><th>model</th><th>mode</th><th>decode all_reduce</th><th>sc2 rule covering it</th><th>stock</th><th>none (DDA)</th><th>nodda</th><th>sc2</th><th>dummy</th>"
                    "<th>sc2 vs nodda</th><th>none (DDA) vs sc2</th><th>stock vs none</th></tr>")
    else:
        page.append("<div class=tw><table><tr><th>model</th><th>mode</th><th>decode all_reduce</th><th>sc2 rule covering it</th><th>stock</th><th>sc2</th><th>none</th><th>dummy</th>"
                    "<th>sc2 vs none</th><th>stock vs sc2</th></tr>")
    for mtag, _ in MODELS:
        for mode, mdd in MODES:
            if (RUN / "nodes.txt").exists() and (mtag, mode) not in NODE:
                continue
            vals = {}
            ok = True
            for p in ("pass1", "pass2"):
                f = RUN / mtag / mode / p / "score.csv"
                if not f.exists():
                    ok = False; break
                for r in csv.DictReader(open(f)):
                    vals.setdefault(r["arm"].split("_")[0], []).append(float(r["median_toks"]))
            if not ok or any(len(vals.get(a, [])) < 2 for a in ARM_ORDER):
                page.append(f"<tr><td>{mtag}</td><td>{mode}</td><td colspan={len(ARM_ORDER) + 5} class=n>not available yet</td></tr>"); continue
            conc = CONC[mode]; b = conc * hidden[mtag] * 2
            rule = next((f"{size_k(r['min_bytes'])}..{size_k(r['max_bytes'])} {r['algorithm']}/{r['protocol']}/{r['channels']}" for r in rules if int(r["min_bytes"]) <= b <= int(r["max_bytes"])), None)
            s2n = [(vals["sc2"][i] / vals[BASE][i] - 1) * 100 for i in range(2)]
            sts = [(vals["stock"][i] / vals["sc2"][i] - 1) * 100 for i in range(2)]
            fmt = lambda a: f"{vals[a][0]:.0f} / {vals[a][1]:.0f}"
            if ROCM10:
                n2s = [(vals["none"][i] / vals["sc2"][i] - 1) * 100 for i in range(2)]
                stn = [(vals["stock"][i] / vals["none"][i] - 1) * 100 for i in range(2)]
                page.append(f"<tr><td>{mtag}</td><td>{mode}</td><td>{size_k(b)}</td><td class={'n' if rule else 'b'}>{rule or gap_label(b)}</td>"
                            f"<td>{fmt('stock')}</td><td>{fmt('none')}</td><td>{fmt('nodda')}</td><td>{fmt('sc2')}</td><td>{fmt('dummy')}</td>"
                            f"<td class={'g' if s2n[0] > 2 else 'n'}>{s2n[0]:+.1f}% / {s2n[1]:+.1f}%</td><td class={'g' if n2s[0] > 2 else 'n'}>{n2s[0]:+.1f}% / {n2s[1]:+.1f}%</td>"
                            f"<td class=g>{stn[0]:+.1f}% / {stn[1]:+.1f}%</td></tr>")
                continue
            page.append(f"<tr><td>{mtag}</td><td>{mode}</td><td>{size_k(b)}</td><td class={'n' if rule else 'b'}>{rule or gap_label(b)}</td>"
                        f"<td>{fmt('stock')}</td><td>{fmt('sc2')}</td><td>{fmt('none')}</td><td>{fmt('dummy')}</td>"
                        f"<td class={'g' if s2n[0] > 2 else 'n'}>{s2n[0]:+.1f}% / {s2n[1]:+.1f}%</td><td class=g>{sts[0]:+.1f}% / {sts[1]:+.1f}%</td></tr>")
    page.append("</table></div>")
    if ROCM10:
        page.append(ORDER_NOTE_ROCM10)
    else:
        page.append("<p class=note>Order in all 16 tables: stock &gt; sc2 &gt; none &gt;&gt; dummy. Stock SGLang (custom all-reduce) stays 8-34% ahead of the tuned RCCL path; "
                    "within the RCCL path the conf is worth +5..+28% where its rules cover the mode's decode all_reduce size, and 0% where they do not (gpt-oss mix).</p>")
    page += conf_section()
    page.append("<h2>2. Results per model, mode and pass</h2>")
    page.append("<div class=tw><table class=fit><tr><th>arm</th><th>meaning</th></tr>" +
                "".join(f"<tr><td>{a}</td><td>{ARMS[a]}</td></tr>" for a in ARM_ORDER) + "</table></div>")
    for mtag, mdesc in MODELS:
        page.append(f"<h2>{html.escape(mdesc)}</h2>")
        for mode, mddesc in MODES:
            if (RUN / "nodes.txt").exists() and (mtag, mode) not in NODE:
                continue  # this model was not run in this mode
            d = RUN / mtag / mode
            page.append(f"<h3>mode {mode} &mdash; {mddesc}</h3>")
            for p in ("pass1", "pass2"):
                page += pass_table(d / p / "score.csv", f"{mtag} · {mode} · {p} · arm order {ORDER[p]} · graphs ON, radix OFF · node {NODE[(mtag, mode)]} · 5 scored reps per arm · <code>{mtag}/{mode}/{p}/score.csv</code>")
    page.append("<h2>3. Which sc2 rules the models hit</h2>")
    page.append("<p class=note>Two views. <b>Decode</b>: with CUDA graphs the decode all_reduce is captured once per batch size and replayed, so its size is fixed by the mode "
                "(concurrency x hidden x 2 bytes; hidden 2048 for Qwen3-30B-A3B, 2880 for gpt-oss-120b" + (", 7168 for DeepSeek-R1" if ROCM10 else "") + ", corroborated by every logged all_reduce size being a multiple of 4096 / 5760 B) "
                "and the rule it lands on is computed below. <b>Detect-server hit counts</b> (TUNING logs, 1 rep) count graph capture (every batch size once) plus prefill calls.</p>")
    hidden = HIDDEN
    rules = [r for r in csv.DictReader(l for l in open(SC2 / "all_reduce_1n.final.conf") if not l.startswith("#"))]
    page.append("<div class=tw><table class=fit><tr><th>model</th><th>mode</th><th>decode all_reduce bytes</th><th>sc2 rule that covers it</th><th>measured sc2 vs " + BASE + " (pass1 / pass2)</th></tr>")
    for mtag, _ in MODELS:
        for mode, mdd in MODES:
            if (RUN / "nodes.txt").exists() and (mtag, mode) not in NODE:
                continue
            conc = CONC[mode]
            b = conc * hidden[mtag] * 2
            hit = next((f"{size_k(r['min_bytes'])}..{size_k(r['max_bytes'])} {r['algorithm']}/{r['protocol']}/{r['channels']}" for r in rules if int(r["min_bytes"]) <= b <= int(r["max_bytes"])), None)
            gains = []
            for p in ("pass1", "pass2"):
                f = RUN / mtag / mode / p / "score.csv"
                if f.exists():
                    rows = {x["arm"].split("_")[0]: x for x in csv.DictReader(open(f))}
                    if "sc2" in rows and BASE in rows and float(rows[BASE]["median_toks"]):
                        gains.append(f"{(float(rows['sc2']['median_toks'])/float(rows[BASE]['median_toks'])-1)*100:+.1f}%")
            cls = " class=v" if hit else " class=x"
            page.append(f"<tr{cls}><td>{mtag}</td><td>{mode} (conc {conc})</td><td>{b:,} ({size_k(b)})</td><td>{hit or gap_label(b) + ' - RCCL default runs'}</td><td>{' / '.join(gains) or '-'}</td></tr>")
    page.append("</table></div>")
    if ROCM10:
        page.append("<p class=note><b>Stock check:</b> a stock server (AITER custom all-reduce on) with the sc2 plugin loaded and TUNING logs, for every model and mode "
                    "(<code>&lt;model&gt;/&lt;mode&gt;/detect/hitmap_stocksc2.csv</code>): 0 rule hits everywhere, one 4-byte all_reduce seen by RCCL per server. In stock SGLang no model all_reduce reaches RCCL. "
                    "<b>DDA check:</b> with RCCL as shipped (arm none, RCCL_DDA_ENABLE default) the tuner is not consulted for the model's all_reduces either: the attempt-1 detect servers and the probe "
                    "(<code>HANDS.md</code>) logged 0 rule hits and one 4-byte all_reduce; the plugin arms therefore run with RCCL_DDA_ENABLE=0.</p>")
    else:
        page.append("<p class=note><b>Stock check:</b> a stock server (custom all-reduce on) with the sc2 plugin loaded and TUNING logs, at p8k and mix for both models "
                    "(<code>&lt;model&gt;/&lt;mode&gt;/stockdetect/hitmap_stocksc2.csv</code>): 0 rule hits, one 4-byte all_reduce seen by RCCL per server. In stock SGLang no model all_reduce reaches RCCL, prefill chunks included.</p>")
    page.append("<p class=note>'share' = share of this conf's rule hits (misses excluded). Counts differ per mode through the prefill chunks; decode capture counts are the same for a model.</p>")
    for mtag, mdesc in MODELS:
        for mode, _ in MODES:
            if (RUN / "nodes.txt").exists() and (mtag, mode) not in NODE:
                continue
            d = RUN / mtag / mode / "detect"
            page += hitmap_table(d / "hitmap_sc2.csv", f"{mtag} · {mode} · arm sc2 · node {NODE[(mtag, mode)]} · <code>{mtag}/{mode}/detect/hitmap_sc2.csv</code>")
            page += hitmap_table(d / "hitmap_dummy.csv", f"{mtag} · {mode} · arm dummy · <code>{mtag}/{mode}/detect/hitmap_dummy.csv</code>")
            if (d / "hitmap_stocksc2.csv").exists():
                page += hitmap_table(d / "hitmap_stocksc2.csv", f"{mtag} · {mode} · STOCK server + sc2 plugin (where does stock's all_reduce go?) · <code>{mtag}/{mode}/detect/hitmap_stocksc2.csv</code>")
    (HERE / OUT).write_text("\n".join(page))
    print("wrote", HERE / OUT)


if __name__ == "__main__":
    main()
