# gpt-oss-120b prefill band UNIT-CORRECTED channel retest (gcf*) — 112/168/224-total rules

2026-09-02, node **amd-mi355x-3** (first tuning run on this node), jobid **20955** (salloc
`ylerman-chanfix-g`, released after). Follow-up to the gp8kv finals
(results-tuning/2026-09-01-6-gptossprefill-v2/, RUNLOG 2026-09-01T21:06Z), where every
prefill-band rule lost badly (best arm ring_simple_48: −5.25% tok/s, **+15.0% TTFT**).

## Question

The gp8kv candidates wrote `channels=16..48` into the conf — but per the channel-unit trap
(findings/INSIGHTS.md §11) a plugin-rule channel value is the **POST-multiplier total**,
while the RCCL default at the dominant 94 MiB size runs **112 actual channels**
(nc:112 / `channel{Lo..Hi}={0..111}`). So the old arms deployed 16–48 actual vs the
default's 112 — pure under-provisioning, not a fair test. Do unit-corrected totals
(112/168/224) tie or beat the default?

## Setup (identical to gp8kv v2 except node + arms)

- Model: `/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a` (verified complete on node 3: config.json + 15 safetensors, 183G resolved)
- Image `lmsysorg/sglang:v0.5.17-rocm720-mi35x` (was missing on node 3 — pulled fresh, ~45 min)
- `IM_EXTRA="--attention-backend triton --disable-radix-cache"`, IM_ISL=8192 IM_OSL=128, conc 32, 128 prompts — prefill-heavy, **TTFT headline**
- Driver `/opt/shared/ylerman/GPU-107/infer-2026-08-30/infer_paired.sh` (one live server per phase, hot-reload plugin, live-conf swap, round-robin)
- Rules: `sw_gcf_ring_simple_{112,168,224}.conf` — single range rule
  `allreduce,33554432,100663296,ring,simple,<ch>,1,8,-1,-1` (same 32–96 MiB band as gp8k)
- Quiet timing (RM_SUBSYS default INIT,ENV); detect mode (INIT,TUNING,ENV) only probe/canary, never timed

## 1. PROBE (gcfprobe, detect, 1 round, rs224 + none) — GO, plus the key mechanism finding

- Regime valid: ServerArgs `disable_radix_cache=True`, all 139 `#cached-token:` lines 0,
  TTFT ~1220/1036 ms (uncached; the invalid cached runs sat at ~257 ms).
- Rank 0: **5037 `Applied config`**, 100% in-band, 100% `ring/simple/channels=224`
  (4380 @ 94,371,840; 365 @ 47,185,920; 73 × four tail sizes) — byte histogram identical
  to the node-2 gp8kv probe.
- **REQUESTED vs ACTUAL — the clamp:** the authoritative final-selection line shows the
  rs224 window ran `AllReduce: 94371840 Bytes -> Algo RING proto SIMPLE
  channel{Lo..Hi}={0..111}` — **112 actual channels, identical to the none window's
  default**. The requested 224 was clamped to the communicator's channel count.
  Cross-check on the node-2 gp8kv probe logs: the rs48 window shows `{0..47}` — the conf
  channel value does reach execution when ≤ the comm's channels, so the clamp is real,
  not a logging artifact. (The per-op `pre/post-adjustment ... nc:112` lines print the
  cost-model value *before* the tuner override lands — `channel{Lo..Hi}` is the line to
  trust, as findings already noted for rccl-tests.)
- **In-model channel ceiling is 112, not the 222–224 seen in bare rccl-tests WarpSpeed
  runs** — the sglang-server communicator is created with 112 channels, so no conf value
  can deploy more than 112 in this workload; rs224 tests parity-at-best, never an
  over-provisioned win.

## 2. FINALS (gcf, quiet, fresh server, 5 arms × 30 rounds, 150/150 reps, 0 fails, 15:27→17:16Z)

| arm | med tok/s (range) | med Mean-TTFT ms | med Mean-TPOT ms | Δ tok/s vs none | Δ TTFT vs none | MWU p (tok/s / TTFT) | paired signs (tok/s, TTFT) |
|---|---|---|---|---|---|---|---|
| **none** (plugin, zero rules) | 684.84 (674–703) | 1036.3 | 38.73 | — | — | — | — |
| current (`all_reduce_1n.final.conf`) | 688.07 | 1040.1 | 38.69 | +0.47% | +0.37% | 0.86 / 0.87 | +15/−15, +18/−12 (tie) |
| **rs112** | **684.90 (656–708)** | **1038.2** | 38.84 | **+0.01%** | **+0.19%** | 0.31 / 0.79 | +17/−13, +15/−15 (**tie**) |
| rs168 | 661.09 | 1166.1 | 39.47 | −3.47% | **+12.53%** | 3e-11 / 4e-10 | 0/30, 29/1 worse |
| rs224 | 657.38 | 1169.2 | 39.60 | −4.01% | **+12.82%** | 3e-11 / 4e-10 | 0/30, 29/1 worse |

Paired per-round means vs none: rs112 +2.41 tok/s (+0.35%), −0.1 ms TTFT; rs168 −24.4
tok/s, +131 ms TTFT; rs224 −25.7 tok/s, +130 ms TTFT; current −0.94 tok/s, +2.5 ms.

### VERDICT — unit fix confirmed, parity at best; over-request actively hurts

1. **rs112 closes the old +15% TTFT gap completely: dead tie with none** (TTFT +0.19%,
   p=0.79; tok/s +0.01%, p=0.31). Writing the post-multiplier total that matches the
   default's actual channel count (112) removes the entire gp8kv penalty. This is also a
   perfect physical A/A (canary: forced ring/simple/112 ≡ default ring/simple/112), and
   it measured 0 — validating the paired design's noise floor.
2. **No rule BEATS the default.** rs112 = parity, nothing more; the two arms above the
   ceiling lose. The default already runs the best expressible config in-model.
3. **rs168/rs224: −3.5%/−4.0% tok/s, +12.5%/+12.8% TTFT, 0/30 rounds above none,
   p≈3e-11** — despite the probe showing the same executed span `{0..111}` and same
   algo/proto as default. Requesting more channels than the communicator owns is NOT a
   harmless clamp: some internal sizing (chunking/steps computed from the raw requested
   count is the leading hypothesis — unobserved in these logs) degrades prefill by ~13%
   TTFT. Never write a channel total above the in-model ceiling.
4. `current` ≡ `none` (A/A control, p=0.86) — fourth consecutive replication.
5. Chain closed: old rs48 loss = genuine channel under-provisioning (48 < 112);
   rs112 = parity; >112 = over-request penalty. The prefill-band recommendation for
   gpt-oss-120b stays **ship no rule** (rule = parity at best, misconfiguration risk on
   both sides of 112, zero upside).

## 3. CANARY (gcfcanary, detect, 1 round, rs112 + none, post-finals) — PASS

- Rank 0: **5037 `Applied config`, all `algo=ring, proto=simple, channels=112`**, byte
  histogram identical to probe; swap rs112→none hot-reloaded at line 363078; **0 Applied
  after**; 94 MiB span `{0..111}` in BOTH windows (physical parity confirmed).
- Driver counted 8 reload events (same init-race quirk as gp8kv: the first empty→rule
  swap logs as `Loaded` without the `hot-reloaded` keyword); finals' `hot-reload
  events: 0` is the known quiet-mode logging artifact — mechanism bracketed by probe
  (before) and canary (after).
- Canary rep (detect, non-timing): rs112 660.86 vs none 680.15 tok/s — first-bench-
  after-startup warmup sits on the rs112 window; quiet finals (n=30) show the true tie.

## Caveats

- Single node (amd-mi355x-3), TP8, one workload shape; absolute numbers not comparable to
  node-2/4/5 runs (node 3 sits ~2% below node 2 on none: 684.8 vs 700.0).
- `--attention-backend triton --disable-radix-cache` as in all corrected prefill runs.
- `none` = hot-reload plugin loaded with zero rules, not plugin-absent.
- An unrelated idle container (`tms-verify`, `sleep infinity`, 0% GPU) existed on the node
  at run start and was removed by its owner during the session; GPUs verified idle before
  and after. The paired round-robin design plus the clean A/A ties (current, rs112) show
  no contamination.
- The rs168/rs224 damage mechanism is inferred from timing + absence of any visible
  config difference in the detect logs; the internal RCCL path that consumes the raw
  requested channel count was not identified here.

## Files

- `gcf_metrics.csv` — arm,round,tps,mean_ttft,median_ttft,mean_tpot (150 rows)
- `analyze.py` — medians + MWU + paired diffs (`python3 analyze.py gcf_metrics.csv`)
- `paired_gcf.log`, `paired_gcfprobe.log`, `paired_gcfcanary.log` — driver logs
- `sw_gcf_ring_simple_{112,168,224}.conf` — the unit-corrected rule arms
- Node 3: `/data/ylerman/models-2026-08-30/gcf{probe,,canary}_*` (all rccl logs gzipped),
  driver stdout `/data/ylerman/gcf*_driver.log`, metrics `/data/ylerman/gcf_metrics.csv`
- Confs shared: `/opt/shared/ylerman/GPU-107/infer-2026-08-30/sw_gcf_*.conf`
