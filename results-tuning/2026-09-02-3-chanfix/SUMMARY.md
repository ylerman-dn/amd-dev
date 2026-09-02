# chanfix — qwen30b prefill retest with UNIT-CORRECTED channel counts (node amd-mi350x-ses2-1, jobid 20952)

2026-09-02, all times UTC. Follow-up to campaign8 (results-tuning/2026-09-01-3-qwenprefill): there every
rule LOST in the 24–96 MiB prefill band, but the swept confs wrote channels 8–48 = **post-x4 totals**
(channel-unit trap, findings/INSIGHTS.md sec. 11) against a default believed to run ~224 total. This
campaign re-arms the same band with properly-provisioned totals 112/168/224.

## Hardware caveat — different SKU than campaign8

- Node: **amd-mi350x-ses2-1** = AMD Instinct **MI350X** (0x75a0, NPS1/SPX, 1000 W cap), gfx950 — same
  ISA as the MI355X nodes used in campaign8, but the lower-clock SKU.
- **Absolute tok/s here is NOT comparable to campaign8's node-2 numbers. Only within-node arm-vs-arm
  comparisons are valid.**

## Setup

- Alloc: `salloc --no-shell -p TEST -N1 -w amd-mi350x-ses2-1 --gres=gpu:8 -t 480 -J ylerman-chanfix`
  → jobid **20952** (created by the earlier stopped attempt, adopted; image
  `lmsysorg/sglang:v0.5.17-rocm720-mi35x` was already pulled by it too).
- Model: hub snapshot `/data/hf_cache/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
  (complete 57 G, 16 shards; the task-named `/huggingface/hub/models--Qwen--Qwen3-30B-A3B` on this node
  is a refs-only stub pointing at the same revision ad44e777).
- Driver: `infer_paired.sh` (single live server, hot-reload tuner plugin, conf swap between reps),
  `IM_ISL=8192 IM_OSL=128 IM_EXTRA="--disable-radix-cache"`, conc 32, 128 prompts — campaign8's
  corrected no-cache regime, identical for all arms.
- Confs `sw_qcf_ring_simple_{112,168,224}.conf`:
  `allreduce,25165824,100663296,ring,simple,<ch>,1,8,-1,-1` (24–96 MiB band from campaign8 detect).

## Phase 1 — PROBE (qcfprobe, detect logging, 1 round, rs224 vs none)

Round tok/s (detect-taxed, non-timing): ring_simple_224 535.57, none 561.97; 16 hot-reload events.

- **Uncached prefill confirmed**: server.log `#cached-token: 0` ×139; no other cached-token values.
- **Rule fires**: 6693 `Applied config` lines per rank (all 8 ranks), 100 %
  `algo=ring, proto=simple, channels=224`, 100 % at in-band bytes
  (67108864 ×5820, 33554432 ×388, five 26–32 MiB mixed-chunk sizes ×97 — matches campaign8 histogram).
- **Requested-vs-actual channels (the campaign question)**: every in-band
  `pre-adjustment`/`post-adjustment ... nBytes:<band> nc:<c>` line in the rs224 segment says
  **nc:112** (13,386/13,386), identical to the none segment (also all nc:112). Across all 8 ranks and
  ALL sizes (decode included) the entire run contains **only nc:112** (1,920,704 lines) — no nc:224,
  no other value.
- Reading: on this node/build the schedule-side channel count is pinned/capped at **112**; a rule
  requesting 224 does not raise it. `nc` is the best channel evidence this build emits
  (channelLo/Hi format absent — same limitation noted 2026-09-01 defprobe).

## Phase 2 — FINALS (qcf, quiet, fresh server, 30 rounds × 5 arms, round-robin)

Arms: none:NONE, current:all_reduce_1n.final.conf, rs112, rs168, rs224.
Server up 15:22, DONE 17:18 (1 h 56 m); 150/150 reps, 0 failures. Driver's `hot-reload events: 0`
line is the known quiet-mode logging artifact (TUNING-gated); mechanism proven by probe + canary.

| arm | med tok/s | d vs none | mean TTFT ms | mean TPOT ms | MWU vs none | paired vs none (mean diff, signs, sign-test p) |
|---|---|---|---|---|---|---|
| current (`all_reduce_1n.final.conf`) | 589.43 | +0.03% | 1066.6 | 46.35 | z=0.14, p=0.89 | +0.44 tok/s, +15/-15, p=1.0 |
| **none** (plugin loaded, zero rules) | 589.27 | -- | 1063.0 | 46.42 | -- | -- |
| **rs112** (unit-corrected) | 588.53 | **-0.13%** | 1061.2 | 46.44 | z=-0.52, **p=0.60 (TIE)** | -0.09 tok/s, +15/-15, p=1.0 |
| rs168 | 563.23 | **-4.42%** | 1238.0 | 47.55 | z=-6.57, p=5.0e-11 | -25.70, +1/-29, p~0 |
| rs224 | 563.17 | **-4.43%** | 1244.2 | 47.55 | z=-6.59, p=4.3e-11 | -26.20, +0/-30, p~0 |

Damage localization mirrors campaign8: losers pay in TTFT (prefill, +16-17%), TPOT nearly flat.

## Phase 3 — CANARY (detect, 1 round, best rule arm = rs112)

`qcfcanary`, detect logging, no-cache, arm rs112 only; round tok/s 554.96 (detect-taxed, non-timing).
**PASS**: 8 hot-reload events (1/rank); 6693 `Applied config` lines on each of 8 ranks, 100 %
`algo=ring, proto=simple, channels=112`, 100 % at exactly the probe/campaign8 in-band sizes
(67108864 x5820, 33554432 x388, five 26-32 MiB sizes x97); `#cached-token: 0` x70; every in-band
post-adjustment line `nc:112` (6693/6693), all nc lines in the whole rank-0 log = 112 (120,142).

## Verdict

**YES - the unit fix closes campaign8's gap, to a tie but not a win.** A properly-provisioned rule
(`ring,simple,112` = the default's actual in-band channel count per nc evidence) is statistically
indistinguishable from no rule: -0.13% median, MWU p=0.60, paired signs +15/-15. Campaign8's -3.46%
for rs48 was therefore the channel-unit trap, not a property of ring/simple itself.

**But over-provisioning actively hurts**: rs168 and rs224 (requests above the node's 112 cap) lose
-4.4% each, unanimously in paired rounds - and land within noise of EACH OTHER (563.23 vs 563.17),
consistent with both clamping to the same effective config plus a fixed penalty for requesting more
than the cap. Note the probe's nc lines show 112 for rs224 too, identical to none - so nc-parity is
NOT performance-parity: the >cap request engages some slower path these logs don't surface
(plausibly forcing plain ring instead of the default's WarpSpeed RING* variant while nc stays 112;
rs112's tie suggests at 112-and-below the tuner path matches the default path).

Practical outcome unchanged from campaign8: **for qwen30b prefill-heavy at TP8 on one node, ship no
rule for the 24-96 MiB band** - the best any correctly-provisioned rule does is tie the default, and
every mis-provisioned variant (under OR over) loses. `current` again ties none (p=0.89): its rules
live in the decode band and don't touch this workload.

SKU caveat: measured on MI350X (gfx950, jobid 20952); campaign8 was MI355X. The tie/loss structure is
within-node evidence; absolute numbers and the 112-channel cap may be SKU/build-specific - the
2026-09-01 defprobe on MI355X node 5 also showed nc:112 at decode sizes, so the "default ~224 total"
belief from the 2026-08-03 env-var sweeps was never confirmed by nc evidence on any node.

## Files

- `sw_qcf_ring_simple_{112,168,224}.conf` — the unit-corrected rule arms.
- `analyze.py`, `fin_stats.csv` — per-round arm,round,tok/s,mean TTFT,mean TPOT + stats.
- Node ses2-1: `/data/ylerman/models-2026-08-30/qcf{probe,}_*` (+ `qcfcanary_*`), drivers' stdout at
  `/data/ylerman/qcf*_driver.log`, rccl detect logs gzipped in place.
