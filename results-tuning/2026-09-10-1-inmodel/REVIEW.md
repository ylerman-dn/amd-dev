# Sub-agent review of the campaign + page (2026-09-10 ~01:15 Israel); actions on top, text below

## Applied
- Page: node in every table title; per-model-mode hit-maps (counts DO differ per mode via prefill chunks); CSV path per table; '4M was parity' corrected (real -1.1% loss); largest uncovered sizes now sorted by size; 'share' defined; hidden-size corroboration stated.
- SUMMARY: pass-agreement precision (31/32 within 0.6%, one -2.2%); drop-first caveat; stock-prefill question marked untested; hidden sizes marked as corroborated, not read.
- Launched the reviewer's closing measurement (stock + sc2 plugin with TUNING logs at p8k and mix, both models) -> stockdetect/ under each mode dir.
- Not done: P(sup) across servers is a different-time pairing (stated in the page intro as "reps of different servers"); config.json not fetched.

## Review text
Review complete. All files read; no edits, no ssh, no cluster.

**1. Recomputation from `bench_rep*.log` (repo `results-tuning/2026-09-10-1-inmodel/`)**
All 64 arm folders re-parsed (drop rep 1, median of 5, gain vs stock median, P(sup) as same-rep wins). Every page number matches the CSVs and the CSVs match the logs, e.g.: Qwen d32 p1 stock 5169.02 / sc2 4168.08 (-19.36%) / none 3513.59 (-32.03%), sc2 vs none +18.63%; Qwen d128 p2 sc2 vs none +21.07%; gpt-oss d128 p1 sc2 16760.89 (-15.93%), none 13139.61 (-34.09%), sc2 vs none +27.56%; gpt-oss mix p1 sc2 7517.73 vs none 7517.66 (+0.00%), p2 -0.06%; gpt-oss p8k p1 stock 1778.21, sc2 1641.28 (-7.70%), none 1557.27 (-12.42%). P(sup)=0.00 for every non-stock arm in all 16 tables (no rep ever overlaps stock).
Drop-first rule: rep 1 is the minimum in only 37 of 64 servers. On node 8 (Qwen d32, p8k) the low rep is **rep 2**, not rep 1, for stock/none/sc2 in both passes (`qwen/d32/pass2/stock_def`: reps 5066, **4918**, 5166, 5165, 5165, 5167 -> rep 2 is -4.8%; `qwen/p8k/pass1/stock_def` rep 2 -5.9%). On node 9 rep 1 is a real cold rep (Qwen mix stock 7257 vs ~7790, -7%). On gpt-oss nodes rep 1 is only -0.7..-1.5%, and the dips are mid-run transients: `gptoss/d128/pass1/stock_def` rep 5 = 18547 (-7.0%), `gptoss/mix/pass2/stock_def` rep 5 = 9362 (-6.7%), `gptoss/d128/pass2/stock_def` has two dips (18532, 18698). Spreads of kept reps: gpt-oss d32/p8k, Qwen mix <=0.8%; Qwen d32/p8k 2-6%; gpt-oss d128/mix 3-8.5%. Medians survive one dip; the two-dip server is the one >2% case below.

**2. Pass agreement (pass2/pass1 per arm)**
31 of 32 arm pairs within 0.6%. One exception: gpt-oss d128 stock -2.23% (19936 -> 19492, `gptoss/d128/pass2/stock_def`, two dipped reps). Consequence: "stock vs sc2" reads -15.9% in pass 1 and -13.9% in pass 2; sc2 vs none is stable (+27.56 / +27.65%). Everything else agrees within the hoped-for ~1%.

**3. Decode-bytes table and the gpt-oss mix gap**
Supported. `gptoss/mix/detect/hitmap_sc2.csv` has `miss,368640,...,219,368640,20x219` and no rule row covers it; sc2 vs none = +0.00% / -0.06% with TPOT 7.60 vs 7.61. Qwen mix (262,144 B) sits on the inclusive upper edge of the 128K..256K rule and is listed in that rule's `sizes_hit` -> +11.5%. The hidden sizes are corroborated by the data itself: every gpt-oss all_reduce size in the hit-maps is a multiple of 5760 (= 2880 x 2 B) and every Qwen size a multiple of 4096; hit counts per rank are 291 = (48 layers x 2 + 1) x 3 for Qwen and 219 = (36 x 2 + 1) x 3 for gpt-oss, consistent with hidden 2048/48 layers and 2880/36 layers. Not confirmed against a primary source: no `config.json` or `server.log` is in the repo (`logs/` dirs are empty, 0 files). Needed: `config.json` from the two snapshot paths in PLAN.md on a node, or the model-config line from a server.log.

**4. Stock arm and prefill**
Nothing in this run can show whether stock used RCCL: pass servers ran with `RM_SUBSYS=INIT,ENV` (chain_template.sh line 25), so no per-call lines exist, and no rccl logs were fetched. The only evidence is 2026-09-09-7 (d32 only): stock+plugin applied 0 rules, one 4-byte all_reduce (`2026-09-09-7-gptoss-attr/SUMMARY.md`). p8k with 64 MB chunks (`qwen/p8k/detect/hitmap_sc2.csv` miss 67108864, 4365 calls) was never probed under stock. What the numbers say: stock's edge at p8k is a decode edge (Qwen TPOT 12.76 vs none 15.84, -19%), while TTFT differs only 950 vs 978 ms (-2.9%); sc2 TTFT equals none (977 vs 978; gpt-oss 1043 vs 1055), as expected with no rule >= 4M. Whether the custom kernel or RCCL+MSCCL carries the 64 MB chunks in stock is undetermined; AGREED's "Risk" item was not tested.

**5. Deviations from AGREED.md**
- Table titles carry mode/pass/order/graphs/radix but **no node** anywhere (only "one MI355X node" in the derivation text); model only via the enclosing h2.
- Hit-maps: AGREED says per model-mode; page shows d32 only per model, asserting counts are identical across modes. They are not: the 2M rule hits 5432 (d32) / 2328 (p8k) / 4656 (d128) / 2328 (mix) in `qwen/*/detect/hitmap_sc2.csv`, and prefill misses differ.
- Hit-map note "largest: 1.9M, 1.9M..." is wrong: misses are sorted by hits, not size; actual largest Qwen miss is 64.9 MB (64872448).
- Conf section says "the 8 sizes >= 4M were parity"; `2026-09-08-2-sanity-check-2/REVIEW.md:4` says 4M is a real -1.1% loss.
- Not traceable to a file: hidden sizes 2048/2880 hard-coded in `build_inmodel_page.py` line 82 (indirectly corroborated by hit-map sizes); pass tables do not cite their CSV individually (one generic path at top); P(sup) pairs reps of servers that ran at different times, not simultaneous rounds.
- "share" column excludes misses (share of rule hits only) - not stated.

**6. Take-aways**
1. sc2 beats plain RCCL on decode in every covered mode: +18.4..18.6% (Qwen d32), +20.9/21.1% (Qwen d128), +27.6% (gpt-oss d128), +11.5% (Qwen mix), +5.2..6.2% (p8k, via TPOT only).
2. Stock still wins everywhere: sc2 is -7.7% (p8k) to -22% (d32) under stock, P(sup)=0.00 in all 16 tables.
3. gpt-oss mix exposes a conf hole: 368,640 B has no rule, sc2 == none (0.0%); the 256K..512K gap costs ~+11..27% there.
4. Reproducibility is excellent: 31/32 arm pairs agree within 0.6%; the one -2.2% is two transient dips on gpt-oss d128 stock.
5. dummy proves the plugin controls the path: -64..-94% tok/s, p8k TTFT 17.7 s (Qwen) / 21 s (gpt-oss).
Next measurement: one stock detect server per model at p8k and mix with `RM_SUBSYS=INIT,TUNING,ENV` (plus COLL if this build needs it) - a single 1-rep server per node, ~15 min, closes the untested "stock falls back to RCCL above the custom kernel's limit" risk. Runner-up: add a 256K..512K rule (searched, not guessed) and rerun gpt-oss mix.
