# gpt-oss-120b PREFILL-HEAVY paired rule sweep v2 (gp8kv*) — corrected no-cache regime

2026-09-01, node **amd-mi355x-2**, jobid **20930**. **Supersedes the node-4 gp8k null**
(results-tuning/2026-09-01-4-gptossprefill/, retracted in RUNLOG 2026-09-01T21:15Z).

## Why a rerun — the radix-cache flaw

The original node-4 gp8k finals were invalid as a prefill test: `bench_serving
--dataset-name random` sends **identical prompts every rep** (fixed seed) and SGLang
RadixAttention serves the prefill from cache after each server's first bench
(`#cached-token: 8191` of 8192). In-band (32–96 MiB) prefill allreduces existed only in
round 1; rounds 2–30 were de-facto no-rule for every arm, so the "no winning rule, all
arms within ~1%" verdict measured only the paired design's noise floor. Discovered by
campaign8 (qwen, same node, RUNLOG 2026-09-01T17:16Z); campaign8's corrected v2 regime —
**`--disable-radix-cache` on the server via `IM_EXTRA`, applied identically to all arms,
bench params unchanged, tools untouched** — is reused here verbatim.

## Setup (fixed facts)

- Model: `/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a` (hub snapshot resolved on node 2)
- `IM_EXTRA="--attention-backend triton --disable-radix-cache"` (triton required — aiter attention crashes gpt-oss; cache-off is the correction)
- Workload: IM_ISL=8192 IM_OSL=128, conc 32, 128 prompts — prefill-heavy, **TTFT is the headline metric**
- Driver: `/opt/shared/ylerman/GPU-107/infer-2026-08-30/infer_paired.sh` (ONE live server per phase, hot-reload plugin `librccl-tunerv4-dn-hotreload.so`, live-conf swap between reps, round-robin arms)
- Rules: `sw_gp8k_*.conf`, single range rule `allreduce,33554432,100663296,<algo>,<proto>,<ch>,1,8,-1,-1` (32–96 MiB band from the node-4 detect histogram — still correct: canary byte histogram is at exactly those sizes)
- Server: sglang v0.5.17-rocm720-mi35x, tp8, `--disable-cuda-graph --disable-custom-all-reduce`, RCCL_MSCCL_ENABLE=0
- Quiet timing (`RM_SUBSYS`=INIT,ENV default); detect mode (INIT,TUNING,ENV) only for probe/canary, never timed

## 1. PROBE (gp8kv_probe, detect, 1 round, ring_simple_48 + none) — regime validation

**GO on all checks:**

- ServerArgs `disable_radix_cache=True`; all 139 `#cached-token:` lines read **0** (both benches).
- **TTFT 1198 ms (ring) / 1032 ms (none) vs the invalid cached run's 257 ms** — prefill is paid every rep.
- Rank 0: **5037 `Applied config`**, 100% in-band, 100% `ring/simple/48`: 4380 @ 94,371,840 (full 16384-tok chunk), 365 @ 47,185,920, 73 each @ 4 tail-chunk sizes.
- Swap to NONE hot-reloaded (log line 363079); **zero Applied after it**, while the none window still contains 44,749 AllReduce ops incl. **13,140 in-band 94,371,840-byte lines** (`No matching config found`) — **uncached in-band prefill allreduces appear in BOTH rounds**, which is precisely what the cached v1 lacked.
- Bonus observation: in the rule-free window RCCL's own choice for the 94 MiB size is **nc:112** channels — far above the conf-expressible max of 48.

## 2. FINALS (gp8kv, quiet, fresh server, 5 arms × 30 rounds, 150/150 reps, ~48 s/rep, 21:06→23:05Z)

Full 30 rounds fit the allocation with ~5 h to spare — no reduction to 20 was needed.

| arm | med tok/s | med Mean-TTFT (ms) | med Mean-TPOT (ms) | Δ tok/s vs none | Δ TTFT vs none | MWU p (both) | paired signs (tok/s, TTFT) |
|---|---|---|---|---|---|---|---|
| **none** (plugin, zero rules) | **700.03** | **1039.4** | **37.88** | — | — | — | — |
| current (`all_reduce_1n.final.conf`) | 699.02 | 1033.0 | 37.92 | −0.14% | −0.61% | 0.53 / 0.43 | +17/−13, +15/−15 (tie) |
| ring_simple_48 | 663.31 | 1195.3 | 39.06 | **−5.25%** | **+15.0%** | 3e-11 | **0/30 above, 30/30 TTFT worse** |
| tree_ll_48 | 547.54 | 1978.5 | 43.22 | **−21.8%** | **+90.4%** | 3e-11 | 0/30, 30/30 |
| tree_ll_16 | 357.87 | 4319.5 | 56.07 | **−48.9%** | **+315.6%** | 3e-11 | 0/30, 30/30 |

Paired per-round means vs none: current −0.92 tok/s (−0.13%), +0.2 ms TTFT;
ring_simple_48 −36.5 tok/s, +174 ms TTFT; tree_ll_48 −151.9, +951 ms; tree_ll_16 −341.2,
+3287 ms. Within-arm spreads are tight (none 687–707; tree_ll_16 354.8–359.1) — no
warmup outliers; the signal is enormous relative to noise.

### VERDICT: NO RULE WINS — and the node-4 "harmless tie" was a cache artifact

- **TTFT headline: every real rule arm makes prefill strictly worse, 30/30 rounds,
  p≈3e-11.** The damage lands exactly where the rules act (TTFT/prefill); TPOT moves far
  less (37.9→39.1 ms for rs48) until channel starvation also drags decode (tree_ll_16 56 ms).
- `current` ≡ `none` (its rules live at 4–1024 KiB and never fire here) — the built-in
  A/A control shows the paired noise floor at ~0.1–0.6%, i.e. the rule penalties are
  50–800× the floor.
- Monotone in channels, ring/simple ≫ tree/ll — same ordering as qwen. RCCL's WarpSpeed
  default uses nc:112 at the dominant 94 MiB size; the tuner conf ceiling is 48 channels,
  so **every forcible combo under-provisions channels and loses**.
- Recommendation unchanged in direction but now with teeth: **ship NO prefill-band
  allreduce rule for gpt-oss-120b** — with real prefill it is not "no benefit", it is
  −5% to −49% throughput and +15% to +316% TTFT.

### Replication of campaign8 (qwen30b, same node/regime)

| | qwen fin2 | gptoss gp8kv (this run) |
|---|---|---|
| none vs current | tie (p=0.97) | tie (p=0.53) |
| best rule arm | ring_simple_48, −3.46% tok/s, +14.7% TTFT, 0/30 | ring_simple_48, −5.25% tok/s, +15.0% TTFT, 0/30 |
| channel trend | fewer ch = worse, monotone | fewer ch = worse, monotone |
| verdict | default WarpSpeed wins | default WarpSpeed wins |

The qwen "rules hurt; default wins" shape **replicates exactly** on a second model family
with a different hidden size (2880 vs 2048), band (32–96 vs 24–96 MiB) and absolute
throughput level. ring_simple_48's TTFT penalty is +15.0% in both — consistent with the
same mechanism (48-ch conf ceiling vs ≥112-ch default) rather than model-specific noise.

## 3. CANARY (gp8kv_canary, detect, 1 round, ring_simple_48 + none, post-finals) — PASS

- **16 hot-reload events** (2 swaps × 8 ranks): `hot-reloaded 2 tuning configurations`
  (empty→rs48) at rank-0 line 1283, `hot-reloaded 1` (rs48→none) at line 363010.
- Rank-0 **5037 `Applied config`, byte histogram IDENTICAL to the probe's**, 100%
  `ring/simple/48`, zero out-of-band; none window: 8760 in-band 94 MiB lines, 0 Applied.
- Independently reproduces the finals penalty: rs48 650.88 vs none 700.68 tok/s;
  TTFT 1237 vs 1031 ms.
- Finals driver line `hot-reload events: 0` is the known quiet-mode logging artifact
  (plugin logs reloads under TUNING subsys only); mechanism is bracketed by probe
  (before) and canary (after) on the same node/driver/plugin/image.

## Caveats

- `--disable-radix-cache` deviates from the frozen server args — it is the sanctioned
  correction (campaign8 precedent) required so the prefill-heavy profile actually
  exercises prefill each rep; identical for all arms, conclusions are within-regime.
- `--attention-backend triton` throughout; not comparable to aiter-backend numbers.
- `none` = hot-reload plugin loaded with zero rules, not plugin-absent.
- Single node (amd-mi355x-2), TP8, one workload shape; absolute numbers not comparable
  across nodes (node-4 cached numbers doubly so).
- Probe quirk (immaterial): in the probe the first swap's reload was logged by a
  second plugin instance as `Loaded 2` without the `hot-reloaded` keyword (init raced
  the swap), so the driver counted 8 events instead of 16; the canary shows the clean
  16. Application correctness was proven by Applied lines in both runs.

## Files

- `gp8kv_metrics.csv` — per-arm per-round tok/s, mean/median TTFT, mean TPOT (150 rows)
- `analyze.py` — medians + MWU (pure-stdlib) + paired diffs; run `python3 analyze.py gp8kv_metrics.csv`
- `paired_gp8kv.log`, `paired_gp8kv_probe.log`, `paired_gp8kv_canary.log` — driver logs
- `sw_gp8k_ring_simple_48.conf`, `sw_gp8k_tree_ll_48.conf`, `sw_gp8k_tree_ll_16.conf` — rule arms
- Node 2: `/data/ylerman/models-2026-08-30/gp8kv{_probe,,_canary}_*` (all rccl logs
  gzipped), drivers' stdout `/data/ylerman/gp8kv*_driver.log`
- Confs shared: `/opt/shared/ylerman/GPU-107/infer-2026-08-30/sw_gp8k_*.conf`
