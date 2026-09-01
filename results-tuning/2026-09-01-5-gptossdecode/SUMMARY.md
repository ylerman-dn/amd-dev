# gpt-oss-120b DECODE paired rule sweep — 184320 B all_reduce (2026-09-01, node 5)

**Question**: gpt-oss-120b (hidden 2880) decodes at concurrency 32 with an all_reduce of
exactly 32×2880×2 = **184320 B every step** — a size matched by NO rule in the current
tuner conf. Does an exact-size rule `allreduce,184320,184320,<algo>,<proto>,<ch>,1,8,-1,-1`
help, and which algo/proto/channels wins?

**Verdict: YES — small but real decode win. `ring,simple,8` is the winner:
median Mean TPOT −0.93% vs no-rule (MWU p=0.0003, better in 25/30 paired rounds),
tok/s +0.45% (p=0.022).** This recovers (and slightly exceeds) the −0.52% quiet-pair
loss the tun arm showed on gpt-oss when its dominant sizes matched nothing
(gptossq, RUNLOG 2026-08-31T10:09Z).

## Setup

- Node: amd-mi355x-5, SLURM job 20927 (`ylerman-prefill8k`, own alloc, held; follow-on
  20931 queued but everything finished under 20927).
- Model: `/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, TP=8.
- Server: sglang v0.5.17-rocm720-mi35x, `--disable-cuda-graph --disable-custom-all-reduce`
  **`--attention-backend triton`** (REQUIRED: aiter backend crashes on gpt-oss hybrid-SWA,
  RUNLOG 2026-08-30T21:52Z).
- Workload: frozen profile — ISL 512 / OSL 512 / concurrency 32 / 128 prompts.
- Driver: `infer_paired.sh` (one live server, hot-reload rule swap between reps,
  round-robin across arms → time drift cancels by construction). Plugin
  `librccl-tunerv4-dn-hotreload.so`, quiet `NCCL_DEBUG_SUBSYS=INIT,ENV` for all timing.
- Phase 1 detect skipped: traffic known analytically (184320 B decode + k×5760 ramp +
  ~2–4 MB prefill at ISL 512); confirmed post-hoc by the canary (zero applied-config
  lines at any size other than 184320 B with the exact rule loaded).

## Phase 2 — Screening (gdecscr, 5 rounds × 15 arms, 16:45→18:34Z, 75/75 reps, 0 fails)

Grid {ring,tree} × {ll,ll128,simple} × {8,16,24,32,48} minus the plugin-IGNOREd
ring_ll128 / tree_ll128 / tree_simple → 15 single-rule confs `sw_gdec_<algo>_<proto>_<ch>.conf`.

5-rep medians (tok/s), ranked:

| arm | median | min | max |
|---|---|---|---|
| ring_ll_8 | 980.09 | 960.45 | 992.53 |
| ring_simple_24 | 974.94 | 892.17 | 985.07 |
| ring_simple_8 | 974.29 | 966.00 | 984.52 |
| tree_ll_16 | 973.12 | 932.86 | 979.24 |
| tree_ll_8 | 972.56 | 930.27 | 986.19 |
| tree_ll_32 | 972.11 | 896.50 | 976.40 |
| ring_simple_48 | 971.92 | 886.98 | 981.19 |
| tree_ll_24 | 971.35 | 932.78 | 977.36 |
| ring_ll_16 | 970.86 | 958.56 | 981.82 |
| ring_ll_48 | 969.94 | 964.03 | 980.27 |
| ring_ll_24 | 969.54 | 949.17 | 981.00 |
| ring_simple_16 | 969.12 | 949.57 | 974.80 |
| ring_ll_32 | 966.92 | 949.87 | 984.90 |
| tree_ll_48 | 965.49 | 951.75 | 980.22 |
| ring_simple_32 | 959.18 | 888.16 | 986.57 |

Spread 2.2% (≈ known drift); top-3 → finals.

## Phase 3 — Finals (gdecfin, 30 rounds × 5 arms, fresh server, 18:36→22:12Z, 150/150 reps, 0 fails)

Arms: top-3 + `none:NONE` (plugin loaded, zero rules) + `current:all_reduce_1n.final.conf`.
30-round medians; **TPOT is the headline metric (decode regime)**:

| arm | median tok/s | mean TTFT (ms) | median Mean TPOT (ms) | TPOT vs none | MWU p (TPOT vs none) |
|---|---|---|---|---|---|
| none | 965.85 | 282.1 | 32.650 | — | — |
| current | 964.58 | 271.9 | 32.560 | −0.28% | 0.367 |
| ring_ll_8 | 969.19 | 287.9 | 32.365 | −0.87% | 0.0022 |
| ring_simple_24 | 968.97 | 273.1 | 32.445 | −0.63% | 0.0133 |
| **ring_simple_8** | **970.17** | **266.1** | **32.345** | **−0.93%** | **0.0003** |

Winner stats (paired per-round diffs, same-minute sampling):

- ring_simple_8 vs none — TPOT: MWU p=0.0003, mean diff −0.242 ms, better 25/30 rounds;
  tok/s: p=0.022, mean diff +7.97, better 25/30.
- ring_simple_8 vs current — TPOT: p=0.0011, mean diff −0.200 ms, better 22/30;
  tok/s: p=0.071, mean diff +5.79, better 20/30.
- All three exact-size arms beat none on TPOT with p<0.02; `current` ≈ `none`
  (expected — current conf has no rule at 184320 B, only consultation overhead).

## Phase 4 — Canary (gdeccanary, 1 detect round, RM_SUBSYS=INIT,TUNING,ENV, 22:14→22:16Z)

**VERIFIED**: **1,196,032** `Applied config ... bytes=184320` lines across the 8 rank
logs, ALL `algo=ring, proto=simple, channels=8`; **0** applied-config lines at any other
size; hot-reload events 8/8 ranks (proves the paired swap mechanism engaged).
Detect logs gzipped in place (`gdeccanary_srv/logs/rccl.*.log.gz`).

## Phase 6 — Profile (prof_gdec, 22:18→22:22Z, separate server, never mixed with timing)

Winner conf loaded statically; server env `SGLANG_TORCH_PROFILER_DIR=/workspace/out/traces`;
bench `--profile --profile-num-steps 25`. Rep tok/s 529.14 (profiler tax — non-timing).
Traces: `node5:/data/ylerman/models-2026-08-30/prof_gdec/traces/1788301224.7741385/`
(8 ranks × ~75 MB gz). Rank-0 comm fraction: **85.2%** of GPU kernel time is
`ncclDevKernel_Generic_1` (743.4 ms of 872.3 ms over the 25 profiled steps) — an UPPER
bound on true comm cost (ring/simple kernels include spin-wait), but it shows decode is
overwhelmingly communication-kernel dominated, which is why a per-size rule moves TPOT.

## Caveats

- NONE arm = plugin-loaded-zero-rules (shared live server keeps the plugin loaded), not
  a no-plugin baseline; quiet campaigns showed plugin presence without matches costs
  nothing measurable.
- `--attention-backend triton` mandatory for gpt-oss on this stack; results are for that
  backend.
- server.log DOES show `Enable Aiter AllReduce Fusion for GptOssForCausalLM` (both
  screening and finals servers) even though server_args prints
  `enable_aiter_allreduce_fusion=False` — enabled by model code. Some gpt-oss all_reduces
  may bypass RCCL, yet the canary proves 1.2M/rep still hit the tuner at 184320 B, and the
  effect is measurable.
- Effect size (~0.9% TPOT) is small vs the session's ~2% drift; credibility rests on the
  paired design (25/30 rounds better) and p=0.0003, not on the raw medians.

## Paths

- Node 5 data: `/data/ylerman/models-2026-08-30/{gdecscr_*,gdecfin_*,gdeccanary_*,prof_gdec}` and
  `paired_gdec{scr,fin,canary}.log`; launchers `launch_gdec{scr,fin,canary}.sh`, `launch_prof_gdec.sh`.
- Confs: `/opt/shared/ylerman/GPU-107/infer-2026-08-30/sw_gdec_*.conf` (15).
- Winner conf copy: `sw_gdec_ring_simple_8.conf` (this dir); per-rep finals metrics: `gdecfin_metrics.csv` (this dir).
- Recommended next step: merge `allreduce,184320,184320,ring,simple,8,1,8,-1,-1` into
  `all_reduce_1n.final.conf` for gpt-oss-class (hidden-2880, conc-32) serving.
