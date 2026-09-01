# campaign7 — prefill-heavy (ISL 8192 / OSL 128) in-model RCCL rule sweep, llama8b, node 5

**Date:** 2026-09-01 (12:48Z–16:40Z) · **Node:** amd-mi355x-5 (SLURM 20917→20927, own allocs) ·
**Model:** Llama-3.1-8B (`/data/mlperf_llama31_8b/model`), TP=8, image `lmsysorg/sglang:v0.5.17-rocm720-mi35x`.

## 1. Workload definition

Same frozen bench harness as campaigns 1–6 except the user-approved workload change:

- **ISL 8192 / OSL 128** (was 512/512), concurrency 32, 128 prompts, `random-range-ratio 1`
- `--disable-custom-all-reduce`, `--disable-cuda-graph`, RCCL_MSCCL_ENABLE=0, stock container RCCL, DN tuner plugin v4 on tun arms
- Driver: `run_many_p8k.sh` — a copy of the shared `run_many.sh` with only ISL/OSL (and container name prefix) changed; the shared script was not touched. Copy archived in this dir.
- ~1.05M prefill tokens + 16,384 output tokens per rep; **output tok/s at OSL 128 is mostly a prefill-wall-time metric**. Quiet rep ≈ 9–10 s; ~6–8 min per arm-run including server start.
- All timing arms quiet (`RM_SUBSYS=INIT,ENV`); logged mode (`INIT,TUNING,ENV`) only for detect + canary (its ~10× throughput tax: 165–181 tok/s vs ~1750 quiet).

## 2. Detect: all_reduce size histogram (rank-0 log, 1 logged rep, current shipped conf)

| nBytes | count | reading |
|---|---|---|
| 262144 | 66,560 | decode (ignore; covered by existing shipped rule) |
| **134217728** | **7,410** | **recurring prefill chunk = 16384 tok × 8192 B** |
| 8192 / 16384 | 3,250 / 1,040 | decode (ignore) |
| **67092480 / 67084288** | **520 / 260** | **recurring prefill tail-chunks (~8190 tok)** |
| 67108864, 66560000, 133570560, 133562368, 133455872, 57344 | ~130 each | one-shot warmup pattern (65 ops × 2 lines) |

**Chosen rule range: 50331648–150994944 (48–144 MiB)** — covers every recurring prefill size with margin; nothing else was observed between 262144 B and 57 MB, so no decode size can leak in.

## 3. Screening (15 valid combos × 5 quiet reps, serial, 12:54Z–14:26Z)

Grid ring/tree × ll/ll128/simple × {8,16,24,32,48} ch minus the probe-proven IGNOREd combos
(ring_ll128, tree_ll128, tree_simple — see RUNLOG 2026-08-31T21:30Z validity map) = 15 arms.

Ranking by 5-rep median tok/s: **tree_ll_48 1810.19**, ring_simple_16 1797.07, tree_ll_24 1779.69,
ring_simple_8 1776.77, tree_ll_32 1772.89, ring_simple_24 1764.66, ring_ll_8 1760.39, ring_ll_24 1760.07,
ring_simple_48 1757.81, ring_simple_32 1756.95, ring_ll_32 1747.30, ring_ll_16 1742.56, ring_ll_48 1735.61,
tree_ll_8 1732.62, tree_ll_16 1727.29.
Spread 4.8 % — far above the rule-effect scale; in hindsight almost entirely node drift (see section 5).

## 4. Finals (top-3 + norule + current, 3 rotated passes × 10 quiet reps/arm, 14:27Z–16:28Z)

150/150 reps, 0 server failures. Medians over all 30 reps per arm:

| arm | median tok/s | delta vs norule | MWU p (vs norule) | median Mean TTFT (ms) | median Mean TPOT (ms) |
|---|---|---|---|---|---|
| norule (def) | 1747.59 | — | — | 283.0 | 16.05 |
| current (`all_reduce_1n.final.conf`) | 1748.46 | +0.05 % | 0.89 | 287.6 | 16.03 |
| tree_ll_48 | 1747.16 | −0.02 % | — | 299.4 | 16.12 |
| **ring_simple_16** (nominal best) | **1755.53** | **+0.45 %** | **0.76** | 276.6 | 16.14 |
| tree_ll_24 | 1755.18 | +0.43 % | — | 282.5 | 16.07 |

MWU = Mann-Whitney U, normal approximation, tie-corrected, two-sided (`mwu.py`, this dir).
Winner-vs-current: ring_simple_16 vs current p=0.59. Above-norule-median counts: ring_simple_16 18/30, tree_ll_24 17/30, tree_ll_48 15/30, current 15/30.

**VERDICT: FLAT — no rule in the 48–144 MiB prefill range produces a significant throughput, TTFT, or TPOT change at this workload.** The screening leader (tree_ll_48, +3.6 % in screen) collapsed to −0.02 % in rotated finals.

## 5. Drift (why the screening ranking is not trustworthy)

Per-pass finals medians (each pass ≈ 40 min): norule 1799.25 / 1760.32 / 1703.01 — a monotonic **−5.3 % slide over the 2 h finals window**, present in every arm. The serial screening (each arm at a different wall-clock time) is therefore confounded at ~5 % scale against a ≤0.5 % effect scale; only the rotated-pass finals design is interpretable. Reproduces the campaign-5/6 drift lesson at larger amplitude.

## 6. Canary verification (phase 4)

1 logged rep, `RM_CONF=sw_p8k_ring_simple_16.conf` (nominal best): **tuner_hits = 35,360** (nonzero, PASS);
every `Applied config` line is `algo=ring, proto=simple, channels=16`; applied-bytes histogram = 134217728 ×29,640, 67092480 ×2,080, 67084288 ×1,040, plus 520 each of 67108864 / 66560000 / 133570560 / 133562368 / 133455872 — **100 % inside the rule range**; 0 `marked as IGNORE` lines. Per-rank op counts match the detect histogram exactly (detect logs 2 lines/op). The finals arms genuinely applied their rules: **the null result is a true null, not a silent no-op.**

## 7. Honest caveats

1. **No winner is claimable.** +0.45 % at p=0.76 is noise; the correct headline is "prefill-sized (64–128 MiB) all_reduce rules do nothing measurable for llama8b at ISL 8192 / OSL 128, conc 32". Plausible mechanism: at these sizes RCCL's default choice is already bandwidth-saturated and the all_reduce is a small slice of prefill compute — but the default combo at 128 MiB was not probed here, so that mechanism is unverified.
2. **rep1 of every server start is a warmup outlier** (~116–181 tok/s, TTFT ~22 s) — 3 of 30 reps per finals arm, 1 of 5 per screening arm. Medians are unaffected; means over these logs would be garbage.
3. **Quiet arms cannot self-verify rule application** (`tuner_hits=0` lines in campaign7.log are expected — no TUNING subsys). Verification rests on the canary (section 6) plus the 2026-08-31 combo-validity probe map; all three finals combos are from the proven-valid set.
4. **Drift dominates screening** (section 5): the top-3 selection is best-effort and combos screened early got a systematic advantage. Since finals put all arms within ±0.5 % of norule, re-screening would almost certainly not change the verdict, but the section-3 ranking should not be reused for anything.
5. Output tok/s with OSL 128 is ~90 % a prefill-latency proxy; no decode-side conclusion can be drawn from this campaign. TTFT/TPOT medians (section 4) are likewise flat.
6. MWU uses the normal approximation (n=30/30, tie- and continuity-corrected) — fine at this n; implementation hand-verified.
7. Detect and canary reps are logging-taxed (~10×) and excluded from all timing claims.

## 8. Artifacts

- Node 5: `/data/ylerman/models-2026-08-30/{p8kdetect_tun,p8kscr_*_tun,p8kfin_*,p8kcanary_tun}` + `campaign7.log`
- Shared dir (`/opt/shared/ylerman/GPU-107/infer-2026-08-30/`): `run_many_p8k.sh`, `run_p8k_screen.sh`, `run_p8k_finals.sh`, `sw_p8k_{ring_ll,ring_simple,tree_ll}_{8,16,24,32,48}.conf`
- This dir: `run_many_p8k.sh`, `run_p8k_screen.sh`, `run_p8k_finals.sh` (deployed copy incl. top-3), `sw_p8k_ring_simple_16.conf` (nominal best), `mwu.py`

## Addendum: comm fraction in this regime (profiled rep, 2026-09-01T16:5xZ)

One profiled rep (same workload, separate server, never mixed with timing): rank-0 kernel-time
split over the 25-step capture = comm 57.9%, attention 11.5%, other 30.5% (GEMMs hide in
unclassified aiter/hipblaslt kernel names — treat the split as coarse). Trace:
node5:/data/ylerman/models-2026-08-30/prof_p8k/traces/. Read together with the flat sweep verdict:
comm is a large share of the prefill window, yet no algo/proto/channel choice moved it — RCCL's
default at 48–144MB appears bandwidth/sync-floor-bound, not routing-bound.
