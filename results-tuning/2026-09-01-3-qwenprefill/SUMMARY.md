# campaign8 — qwen30b PREFILL-HEAVY paired rule sweep (node amd-mi355x-2, jobid 20930)

2026-09-01, all times UTC. Model: `/data/ylerman/qwen30b_model` (local complete copy, hidden_size 2048 → 4096 B/token allreduce payload at TP8/bf16). Image `lmsysorg/sglang:v0.5.17-rocm720-mi35x`, paired driver `infer_paired.sh` (single live server, hot-reload tuner plugin `librccl-tunerv4-dn-hotreload.so`, conf swap between reps).

## Workload (user-approved profile)

- IM_ISL=8192, IM_OSL=128, concurrency 32, 128 prompts (bench params frozen).
- Quiet logging (`NCCL_DEBUG_SUBSYS=INIT,ENV`) for all timed runs; `INIT,TUNING,ENV` only for detect/canary reps.

## Phase 1 — detect histogram and range

One logged rep (`qp8kdetect_tun`, `all_reduce_1n.final.conf`, 162.52 tok/s — non-timing). Rank-0 nBytes histogram:

| nBytes | count | classification |
|---|---|---|
| 131072 | 99328 | decode (32-seq batch × 4096 B) — excluded |
| 67108864 | 11640 | **recurring prefill**: 16384 tok × 4096 B, 60 passes × 194/pass |
| 4096 / 8192 | 4850 / 1552 | decode/small — excluded |
| 9723904 | 1024 | below band, not per-pass signature (1024 ≠ n×194) — excluded |
| 33554432 | 582 | **prefill**: 8192 tok, 3 passes |
| 26497024–31875072 (×5) | 194 each | **prefill**: one-pass mixed-batch chunks |
| 24576 | 194 | one tail chunk — excluded |

Total prefill tokens accounted ≈ 128 prompts × 8192 — checks out.

**Range picked: `min_bytes=25165824, max_bytes=100663296` (24–96 MiB)** — covers all recurring prefill sizes with headroom, excludes decode and the 9.3 MiB anomaly. Rule template (15 confs `sw_qp8k_{ring_ll,ring_simple,tree_ll}_{8,16,24,32,48}.conf`, ll128 & tree_simple combos excluded as plugin-IGNOREd):

```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
allreduce,25165824,100663296,<algo>,<proto>,<ch>,1,8,-1,-1
```

## The v1 sweep was invalid — radix cache absorbed the prefill (CRITICAL finding)

`bench_serving --dataset-name random` generates **identical prompts every rep** (fixed seed), and SGLang RadixAttention caches them: server.log shows `#cached-token: 8191` (of 8192) for every request after a server's first bench. Canary2's round-2 segment contains **zero allreduces ≥ 20 MiB**. Consequence: **in-range ops only exist in each server's FIRST bench**; from round 2 on, every arm is a de-facto no-rule arm.

- Screening v1 (`qp8kscr`, 75/75 reps, medians 756–777) and Finals v1 (`qp8kfin`, 150/150) therefore measured nothing but the paired design's noise floor.
- The round-1 throughput dips (~146 tok/s vs ~770 later) previously read as "warmup" are actually the one real-prefill round.
- This also implicates the llama campaign7 p8k screening on node 5 (one server per arm, reps 2–5 cached) — flagged for re-adjudication, not judged here.

### Finals v1 table (qp8kfin, 30 rounds/arm, cached regime → de-facto A/A; kept as noise-floor evidence)

| arm | med tok/s | Δ vs none | med TTFT ms | med TPOT ms | MWU p vs none |
|---|---|---|---|---|---|
| tree_ll_24 | 773.06 | +0.72% | 248.7 | 39.70 | 0.10 |
| current | 771.54 | +0.53% | 246.6 | 39.78 | — |
| tree_ll_48 | 769.85 | +0.30% | 245.5 | 39.80 | — |
| ring_simple_48 | 768.96 | +0.19% | 245.6 | 39.95 | — |
| none | 767.51 | — | 246.5 | 39.95 | — |

Five "different" arms within ±0.72%, best p=0.10, TTFT flat at ~246 ms (cached prefill) → the paired noise floor is ≲0.8% here. Paired tree_ll_24-minus-none (excl. the real-prefill round 1): mean +5.1 tok/s (+0.67%), signs +19/−10, sign-test p=0.095.

## Corrected regime: `IM_EXTRA=--disable-radix-cache`

Driver's sanctioned env knob (server flag, applied identically to all arms; bench params unchanged; tools untouched). Validation (`qp8kval`, detect, 2 rounds, tree_ll_24): both rounds 389 tok/s, **6693 Applied per round**, 8 hot-reloaded events at the mid-run swap → rules fire every round. GO.

## Phase 2 — screening v2 (qp8kscr2, 15 arms × 5 rounds, no-cache)

| rank | arm | median tok/s | | rank | arm | median tok/s |
|---|---|---|---|---|---|---|
| 1 | **ring_simple_48** | **578.66** | | 9 | ring_ll_24 | 411.69 |
| 2 | **ring_simple_32** | **555.87** | | 10 | tree_ll_24 | 401.44 |
| 3 | **ring_simple_24** | **532.18** | | 11 | ring_simple_8 | 395.90 |
| 4 | ring_ll_48 | 501.79 | | 12 | ring_ll_16 | 350.33 |
| 5 | tree_ll_48 | 492.69 | | 13 | tree_ll_16 | 338.88 |
| 6 | ring_simple_16 | 489.25 | | 14 | ring_ll_8 | 242.97 |
| 7 | ring_ll_32 | 453.07 | | 15 | tree_ll_8 | 231.00 |
| 8 | tree_ll_32 | 441.97 | | | | |

Real signal at last: 2.5× spread, monotone in channels, within-arm spread ~±1%. The v1 "winner" tree_ll_24 ranks 10/15 — v1's ranking was noise.

## Phase 3 — FINALS v2 (qp8kfin2, 30 rounds/arm, no-cache) — THE RESULT

| arm | med tok/s | Δ vs none | med TTFT ms | med TPOT ms | MWU vs none | paired vs none (mean diff, signs) |
|---|---|---|---|---|---|---|
| current (`all_reduce_1n.final.conf`) | 603.45 | +0.12% | 976.1 | 45.66 | p=0.97 | −0.37 tok/s (−0.06%), +13/−17 |
| **none** (plugin loaded, zero rules) | 602.74 | — | 976.2 | 45.77 | — | — |
| ring_simple_48 | 581.87 | **−3.46%** | 1119.9 | 46.57 | z=−6.65, p=2.9e-11 | −22.18 (−3.68%), **+0/−30** |
| ring_simple_32 | 555.53 | **−7.83%** | 1289.3 | 47.83 | p=2.9e-11 | −47.60 (−7.90%), +0/−30 |
| ring_simple_24 | 532.22 | **−11.70%** | 1464.2 | 49.02 | p=2.9e-11 | −70.97 (−11.77%), +0/−30 |

**VERDICT: for the qwen30b prefill-heavy workload, the best tuner rule for the 24–96 MiB allreduce band is NO rule.** Every forcible combo loses unanimously (0/30 paired rounds above none); the damage lands exactly where the rules act — TTFT (prefill) +15%…+50%, TPOT nearly flat. RCCL's default for ≥64 MiB is the WarpSpeed `RING*`/SIMPLE path at 56–224 channels (2026-08-03-3 defaults batch), which tuner confs cannot express (channels ≤48, algo=ring ≠ RING*). The trend line (24→32→48 ch keeps improving, still below default) is consistent with the default's higher channel counts winning. `current` ≈ `none` (p=0.97): its rules live at 4–1024 KiB (decode band) and neither help nor hurt this workload measurably.

## Phase 4 — canary + hot-reload verification

- **canary3** (`qp8kcanary3`, detect, no-cache, ring_simple_48 + none, 2 rounds): rank-0 has **13386 `Applied config` lines (6693 × 2 rounds)**, 100% `algo=ring, proto=simple, channels=48`, 100% at in-range bytes (67108864 ×11640, 33554432 ×776, five 26–32 MiB sizes ×194); **32 `hot-reloaded` events** (4/rank) — mid-serving conf swap proven. Canary independently reproduces the finals penalty (rs48 550.7/572.5 vs none 589.8/604.3).
- **canary1** (`qp8kcanary`, detect, cached, tree_ll_24 only): 6596 Applied at in-range bytes — the v1 finals' rules did fire in round 1.
- **canary2** (`qp8kcanary2`, detect, cached, tree_ll_24 + none): 24 hot-reloaded events; exposed the radix-cache flaw (round-2 segment has zero ≥20 MiB ops).
- **Finals `qp8kfin`/`qp8kfin2` srv logs show 0 "hot-reloaded" lines — this is EXPECTED, not breakage**: the plugin logs only under the TUNING subsys, absent in quiet mode. Verified empirically: qp8kscr srv logs have 12.6M lines/rank, plugin confirmed loaded, 75 swaps, zero plugin lines. Quiet runs are bracketed by detect-mode proof instead: qsanity (14:15Z, PASS, 24 events) before, canary2/val/canary3 after, all on this node/driver/plugin/image.

## Not node drift

The fin1→fin2 absolute drop (~770 → ~530-600 tok/s, "~25%") is the regime change (`--disable-radix-cache` makes every rep pay ~1.05M tokens of prefill), not node degradation: same-arm medians are stable across hours in the same regime (scr2 ring_simple_48 578.66 at ~18:00Z; fin2 rounds 16/29 at 578.2/583.0 around 20:00Z), and none/current sit at ~603 all evening. TTFT (976 ms vs 246 ms cached) confirms the mechanism.

## Caveats

- Model is a local dereferenced copy of the qwen30b weights (complete, 16 shards + config), not the hub snapshot path used by earlier campaigns.
- `none` = hot-reload plugin loaded with zero rules, **not** no-plugin. (Earlier quiet campaigns showed plugin-present-no-match costs nothing measurable.)
- Single node (amd-mi355x-2), single image, TP8, one workload shape; `--disable-cuda-graph --disable-custom-all-reduce`, MSCCL off — absolute numbers are not comparable across nodes or to cuda-graph setups.
- `--disable-radix-cache` is itself a deviation from the frozen server args — required to make the approved prefill-heavy profile actually exercise prefill per rep. All arms shared it; conclusions are within-regime.
- v1 artifacts (`qp8kscr_*`, `qp8kfin_*`) remain on node 2 as the noise-floor/A-A dataset; all rccl srv logs gzipped in place.
- The llama campaign7 p8k screening (2026-09-01, node 5) shares the cached-prefill flaw and needs re-adjudication.

## Files

- `sw_qp8k_ring_simple_48.conf` / `_32` / `_24` — finals v2 rule arms (winner-among-rules = rs48; overall winner = no rule). `sw_qp8k_tree_ll_24.conf` — v1 nominal winner, kept for the record.
- `fin1_stats.csv`, `fin2_stats.csv` — per-round arm,round,tok/s,TTFT,TPOT.
- `screening_v1_paired.log`, `screening_v2_paired.log`, `finals_v1_paired.log`, `finals_v2_paired.log` — driver logs.
- Node 2: `/data/ylerman/models-2026-08-30/qp8k{detect,scr,fin,canary,canary2,val,scr2,fin2,canary3}*`, drivers' stdout at `/data/ylerman/qp8k*_driver.log`.
