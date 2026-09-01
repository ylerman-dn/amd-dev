# gpt-oss-120b PREFILL-HEAVY paired rule sweep (gp8k*) — 2026-09-01, node amd-mi355x-4, job 20929

Mirror of the qwen/llama prefill flows, run entirely with the paired driver
(`infer_paired.sh`, ONE live server per phase, hot-reload rule swaps between reps).

## Setup (fixed facts)

- Model: `/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a` (15 shards, MXFP4)
- hidden_size 2880 → allreduce sizes are multiples of 5760 B (bf16)
- Workload: `IM_ISL=8192 IM_OSL=128`, conc 32, 128 prompts (prefill-heavy; TTFT is the headline metric)
- REQUIRED server extra: `IM_EXTRA="--attention-backend triton"` (aiter attention backend crashes this model)
- Server: sglang v0.5.17-rocm720-mi35x, tp8, `--disable-cuda-graph --disable-custom-all-reduce`, RCCL_MSCCL_ENABLE=0, `chunked_prefill_size=16384`
- Tools: `/opt/shared/ylerman/GPU-107/infer-2026-08-30/{infer_many.sh,infer_paired.sh}`, hot-reload plugin `librccl-tunerv4-dn-hotreload.so`
- Outputs: node4 `/data/ylerman/models-2026-08-30/gp8k*`

## 1. DETECT (gp8kdetect, logged run, RM_CONF=all_reduce_1n.final.conf)

rep1 tok/s=659.34 TPOT=39.24 (detect-mode numbers are NOT timing data — massive logging tax).
`tuner_hits=0`: **none of the shipped conf's exact sizes occur for gpt-oss at this workload.**

nBytes histogram (rank 0, per-forward-pass call-site count = 73):

| nBytes | tokens (÷5760) | count | class |
|---|---|---|---|
| 4 | — | 1 | startup one-shot |
| 5,760 | 1 | 2,993 | decode/warmup single-token |
| 34,560 | 6 | 73 | small odd batch |
| 184,320 | 32 | 37,376 | decode @ conc 32 (dominant decode) |
| 12,869,632 | not ×5760 | 512 | different call site — excluded |
| 37,192,320 | 6,457 | 73 | prefill tail chunk |
| 40,561,920 | 7,042 | 73 | prefill tail chunk |
| 40,659,840 | 7,059 | 73 | prefill tail chunk |
| 43,470,720 | 7,547 | 73 | prefill tail chunk |
| 47,180,160 | 8,191 | 73 | prefill ~1 prompt |
| 47,185,920 | 8,192 | 219 | prefill 1 prompt |
| 94,371,840 | 16,384 | 4,380 | **prefill full chunk (2 prompts) — dominant** |

**Covering rule range picked: 33,554,432–100,663,296 (32–96 MiB)** — covers all recurring
prefill sizes with margin, excludes decode and the non-×5760 call site.

## 2. SCREENING (gp8kscr, paired, one server, 15 arms × 5 rounds, quiet)

Grid: {ring,tree} × {ll,simple} × {8,16,24,32,48} minus plugin-IGNOREd combos
(ring_ll128, tree_ll128, tree_simple — probe-proven 2026-08-31) = 15 confs
`sw_gp8k_<algo>_<proto>_<ch>.conf`, each one range rule `allreduce,33554432,100663296,<a>,<p>,<ch>,1,8,-1,-1`.

Median tok/s over 5 rounds (top-8): tree_ll_16 **921.73**, ring_ll_48 **920.45**, tree_ll_8 **920.39**,
ring_simple_48 919.63, tree_ll_48 919.18, ring_simple_16 918.54, ring_ll_24 918.48, ring_simple_24 918.25.
Bottom: ring_simple_32 909.86. Full spread 910–922 (~1.3%); full data in `gp8kscr_tps.txt`.
(One hiccup rep: ring_ll_8 round-3 249.30 tok/s — stall, median-robust.)

Top-3 → finals: **tree_ll_16, ring_ll_48, tree_ll_8**.

## 3. FINALS (gp8kfin, paired, fresh server, 5 arms × 30 rounds, quiet)

Arms round-robin per round: tree_ll_16, none(NONE), ring_ll_48, current(all_reduce_1n.final.conf), tree_ll_8.

| arm | med tok/s | med TTFT ms | mean TTFT ms | med TPOT ms | TTFT vs none | p (MWU) | tok/s vs none | p |
|---|---|---|---|---|---|---|---|---|
| **none** (plugin, zero rules) | **913.99** | **256.95** | 270.07 | **33.14** | — | — | — | — |
| current | 911.20 | 260.08 | 270.34 | 33.19 | +1.22% | 0.048 | −0.31% | 0.29 |
| tree_ll_16 | 910.48 | 256.32 | 415.57¹ | 33.17 | −0.24% | 0.91 | −0.38% | 0.23 |
| ring_ll_48 | 911.99 | 259.29 | 272.90 | 33.20 | +0.91% | 0.042 | −0.22% | 0.29 |
| tree_ll_8 | 912.52 | 258.94 | 275.64 | 33.19 | +0.78% | 0.096 | −0.16% | 0.71 |

¹ tree_ll_16 mean TTFT wrecked by round 1 (TTFT 4471 ms) — first-ever bench rep on the fresh
server (warmup artifact; it was first in round-robin order). All arms show occasional
320–490 ms TTFT spike rounds (shared server-level noise); medians are the robust statistic.

Paired per-round diffs vs none (sign counts, n=30): current TTFT +4.00 ms median (22 worse/8 better);
ring_ll_48 TTFT +2.50 ms (22/8); tree_ll_8 TTFT +3.52 ms (23/7); tree_ll_16 TTFT +0.18 ms (16/14, tie).
tok/s: all rule arms 18–22 of 30 rounds below none.

### VERDICT: NO WINNING RULE — no-rule is best in this regime

- No rule arm beats `none` on any metric. TTFT (headline): tree_ll_16 statistically ties none
  (p=0.91); ring_ll_48, tree_ll_8 and current are ~1% WORSE (p≈0.04–0.10).
- **Internal noise-floor control**: the `current` arm is functionally identical to `none` for
  gpt-oss (detect proved its rules never fire at these sizes), yet measured +1.22% TTFT
  (p=0.048) → paired-design residual noise floor is ~1% on TTFT at n=30; the marginally
  significant regressions above are at/below that floor. Deltas under ~1% here are not
  actionable either way.
- Recommendation: ship NO prefill-range allreduce rule for gpt-oss-120b at this workload.
  `sw_gp8k_tree_ll_16.conf` (copied here) is the best-performing rule arm (ties none, never
  beats it) — kept only as the sweep's nominal top rule, NOT recommended for deployment.

## 4. VERIFY

- Pre-finals probe (gp8kprobe, detect-mode paired, ring_ll_32 + NONE, 1 round): 16 hot-reload
  events, 39,712 `Applied config` all at in-range prefill sizes with the requested combo.
  Also showed detect-mode timing distortion (probe round 482 vs none 899 tok/s) —
  reconfirming "never time in detect mode".
- **Canary (gp8kcanary, detect-mode paired, winner tree_ll_16, 1 round): PASS** — 8/8 ranks
  `hot-reloaded 2 tuning configurations from live_gp8kcanary.conf`; 39,712 `Applied config`,
  ALL at exactly the 7 recurring prefill sizes from the detect histogram (584 each at the five
  ~37–47 MB tail sizes, 1,752 @ 47,185,920, 35,040 @ 94,371,840), ALL `algo=tree, proto=ll,
  channels=16`; zero out-of-range applications.
- Hot-reload lines in gp8kfin_srv/gp8kscr_srv logs: **0 — this is a logging artifact, not a
  failure.** The plugin emits `hot-reloaded` under the TUNING debug subsys, which quiet runs
  (RM_SUBSYS=INIT,ENV) exclude by design. Mechanism validity for the quiet runs is established
  by the bracketing detect-mode paired runs (gp8kprobe before finals, gp8kcanary after), which
  use the identical driver/plugin/live-conf path on the same node and show reload + application
  on every swap. (Same 0 was already observed in screening and predicted for finals.)

## Caveats

- `--attention-backend triton` throughout (aiter attention backend crashes gpt-oss); numbers
  are not comparable to aiter-backend runs.
- Server log prints `Enable Aiter AllReduce Fusion for GptOssForCausalLM` even with the triton
  backend, BUT ServerArgs shows `enable_aiter_allreduce_fusion=False`, and RCCL demonstrably
  receives the allreduces (39,712 tuner applications per rep in canary; ~19M consult lines per
  rank per run) — the log line is a model-init banner, not an active fusion.
- `none` arm = plugin loaded with zero rules (shared live server), not plugin-absent.
- Even quiet runs (INIT,ENV) log ~19M per-op `pre/post-adjustment`/`minNChannels` lines per
  rank (RCCL core, not subsys-gated as expected): absolute numbers carry a logging tax, but it
  is symmetric across arms (one shared server) so paired deltas remain valid.
- Node-local single-node (1n8r) result; do not compare raw numbers across nodes.

## Files

- `gp8kfin_metrics.csv` — finals per-arm per-round tok/s, mean TTFT, mean TPOT
- `gp8kscr_tps.txt` — screening per-arm per-round tok/s
- `sw_gp8k_tree_ll_16.conf` — nominal top rule arm (NOT recommended; no-rule wins)
- Node 4: `/data/ylerman/models-2026-08-30/{gp8kdetect_tun,gp8kscr_*,gp8kfin_*,gp8kprobe_*,gp8kcanary_*}`,
  driver logs `paired_gp8k{scr,fin,probe,canary}.log`; all rccl logs gzipped.
- Confs on node: `/opt/shared/ylerman/GPU-107/infer-2026-08-30/sw_gp8k_*.conf`
