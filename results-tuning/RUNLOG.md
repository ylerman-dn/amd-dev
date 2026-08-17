# GPU-107 tuning RUNLOG

One row per run, appended at the time of the run. `nodelist` is the actual allocated nodes.
A run that is not here did not happen.

Timestamps are the log-completion time (UTC, from log mtime); durations are mtime deltas
within a batch, so the first run of a batch is approximate.

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-03T09:09:23Z | ~10 | 14048 | amd-mi355x-4 | 1 | `all_reduce_perf -b 256M -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -M 1 -R 1` via `srun --mpi=pmix --ntasks-per-node=8` | `LD_PRELOAD=$MY_PATH/librccl.so`, `NCCL_DEBUG=VERSION`, `HSA_NO_SCRATCH_RECLAIM=1` | `results-tuning/2026-08-03-1-mcheck/A_default.log` | No algo/proto/nchannels columns. `-M` on this build is `--memory_report`, not the algo/proto flag; `-R` is `--local_register`. Both flags in the sweep are wrong. |
| 2026-08-03T09:10:19Z | ~8 | 14048 | amd-mi355x-4 | 1 | same but `-A 1`, no `-M`/`-R` | as above | `results-tuning/2026-08-03-1-mcheck/A2_default.log` | Columns present with `-A 1`. RCCL default = `RING*` / `SIMPLE` / 56 channels. |
| 2026-08-03T09:10:27Z | 8 | 14048 | amd-mi355x-4 | 1 | same, `-A 1` | + `NCCL_ALGO=RING`, `NCCL_PROTO=SIMPLE`, `NCCL_MIN_NCHANNELS=32`, `NCCL_MAX_NCHANNELS=32` | `results-tuning/2026-08-03-1-mcheck/B_ring_simple_32ch.log` | Request **honoured**: reports `RING*`/`SIMPLE`/32. |
| 2026-08-03T09:10:38Z | 11 | 14048 | amd-mi355x-4 | 1 | same, `-A 1` | + `NCCL_ALGO=TREE`, `NCCL_PROTO=LL128` | `results-tuning/2026-08-03-1-mcheck/C_tree_ll128.log` | Request **silently substituted**: reports `RING*`/`SIMPLE`/56, identical to default. Substitution is real and now observable. |

Purpose of this batch: settle whether the sweep can verify which algo/proto/nchannels RCCL
actually selected. Answer: yes, via `-A 1`. No performance claims are made from these runs.

## Batch 2 — algo × proto × nchannels matrix

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-03T09:19:57Z | 195 (19 runs) | 14048 | amd-mi355x-4 | 1 | `all_reduce_perf -b 256M -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1` via `srun --mpi=pmix --ntasks-per-node=8`, once per combo | `NCCL_ALGO` ∈ {RING,TREE} × `NCCL_PROTO` ∈ {SIMPLE,LL128,LL} × `NCCL_MIN/MAX_NCHANNELS` ∈ {19,32,unset}, + 1 unforced reference. Plus `LD_PRELOAD=$MY_PATH/librccl.so`, `NCCL_DEBUG=VERSION`, `HSA_NO_SCRATCH_RECLAIM=1` | `results-tuning/2026-08-03-2-combomatrix/` (19 logs + `SUMMARY.md`) | 19/19 rc=0, `#wrong`=0. Only 3 of 6 algo/proto pairs honoured (RING/SIMPLE, RING/LL, TREE/LL); LL128 always → SIMPLE, TREE+SIMPLE → RING. nchannels always honoured exactly, incl. 19. Half the forced matrix would be mislabelled by the sweep. `unsupported_combos.yaml` has no LL128 rule. |

Per-combo verdicts, the `unsupported_combos.yaml` cross-check, and the `RING*` mapping problem
are in `results-tuning/2026-08-03-2-combomatrix/SUMMARY.md`. Throughput numbers in that file are
single-shot and explicitly not claims.

## Batch 3 — RCCL defaults across 4K–512M (baseline)

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-03T09:53:54Z | 34 (3 reps × ~11s) | 14048 | amd-mi355x-4 | 1 | `all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1` via `srun --mpi=pmix --ntasks-per-node=8`, ×3 | **none forced** (no `NCCL_ALGO`/`NCCL_PROTO`/`NCCL_*_NCHANNELS`). Only `LD_PRELOAD=$MY_PATH/librccl.so`, `NCCL_DEBUG=VERSION`, `HSA_NO_SCRATCH_RECLAIM=1` | `results-tuning/2026-08-03-3-defaults/` (3 logs + `SUMMARY.md`) | 3/3 rc=0, 18 sizes, `#wrong`=0, **0 selection mismatches across repeats**. Defaults: TREE/LL 4K–256K (1→16 ch), RING/SIMPLE 512K–32M (16→56 ch), `RING*` (WarpSpeed)/SIMPLE 64M–512M (56 ch). Spread 0.2–3.1%. |

This batch is the **default baseline** for later A/B. Median-of-3 busbw per size, spread
recorded. Transitions, determinism check and tuning hypotheses in
`results-tuning/2026-08-03-3-defaults/SUMMARY.md`.

Library pinned for this and all subsequent batches: `librccl.so` = RCCL **2.28.3-develop:2e42aa8**
at `/opt/shared/ylerman/GPU-107/bin/librccl.so.1.0`, md5 `53a5c003a247a1d0a94851855037ad85`,
paired with rccl-tests `2.17.9-develop:2e42aa8` from the same commit. This is the newest of the
three librccl builds on the node (system ROCm 7.2.4 ships 2.27.7; `rccl-drop2025-08/` is 2.26.6).

## Batch 4 — NCCL_DEBUG=INFO diagnostic (no throughput numbers)

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-03T11:16:06Z | 11 | 14048 | amd-mi355x-4 | 1 | `all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 5 -w 1 -c 1 -A 1` via `srun --mpi=pmix --ntasks-per-node=8` | `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH,ENV`, `LD_PRELOAD=$MY_PATH/librccl.so`, `HSA_NO_SCRATCH_RECLAIM=1`. Nothing forced. | `results-tuning/2026-08-03-4-infocap/` (`info.log` + `SUMMARY.md`) | **Diagnostic only — no busbw claims.** 18 rows, `#wrong`=0. `-A 1`'s nchannels column proven unreliable (says 56 where real span was 52/54, and 56 where WarpSpeed ran 222–224); algo/proto column confirmed correct. WarpSpeed AR threshold = 67108864 B (64 MiB), engages at 4 of 18 sizes. 48-vs-56 **settled**: real channel counts are 52–224, all above 48, so no cap is in effect. Only the cap's trigger condition remains open (needs a 2-node run). |

Corrections this batch forced: the `nch` column in
`results-tuning/2026-08-03-3-defaults/SUMMARY.md` was wrong and is now shown as both `nch(-A)` and
`nch(real)`. Verification rules in `CLAUDE.md` updated to record that `-A 1` cannot be trusted
for channel counts.

## Batch 5 — does NCCL_DEBUG=INFO distort busbw? (3-arm controlled test)

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-03T12:24:30Z | 172 (15 runs) | 14048 | amd-mi355x-4 | 1 | `all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1 -Z csv -X <file>` via `srun --mpi=pmix --ntasks-per-node=8`; 3 arms × 5 repeats, interleaved round-robin | Arm A `NCCL_DEBUG=VERSION`; Arm B `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH,ENV` → stdout; Arm C same + `NCCL_DEBUG_FILE=..._%p.log`. All: `LD_PRELOAD=$MY_PATH/librccl.so`, `HSA_NO_SCRATCH_RECLAIM=1`. Nothing forced. | `results-tuning/2026-08-03-5-lognoise/` (15 CSVs, 15 logs, `armC_debug_files.tgz`, `SUMMARY.md`) | 15/15 rc=0, `#wrong`=0. **INFO to a file is free** (arm C within ±1.4% of A, ranges overlap 18/18). **INFO to stdout costs a few %** (arm B slower 18/18 sizes: −4.0% at 8K, −1.9% at 1M, ≤0.7% ≥16M, −0.0% at 512M). Cost is stdout contention, not logging. Corrects my earlier overstated "INFO distorts timing" claim → finding 09. |

`-Z csv -X` (rccl-tests' own CSV reporter) was used for all three arms so arm B's interleaved
stdout could not corrupt its own measurements. Verified both INFO arms emitted identical volume
(51,551 lines each) — only the destination differed.

## Batch 6 — acceptance map per size + exotic-algo probes

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-03T12:5xZ | ~140 (11 runs) | 14048 | amd-mi355x-4 | 1 | `all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1 -Z csv -X <file>` via `srun --mpi=pmix --ntasks-per-node=8` | `NCCL_ALGO`×`NCCL_PROTO` for 6 pairs + 4 probes (NVLS, PAT, COLLNET_DIRECT, COLLNET_CHAIN); channels unset. All: `NCCL_DEBUG=INFO` + `NCCL_DEBUG_FILE=..._%p.log`, `LD_PRELOAD=$MY_PATH/librccl.so`, `HSA_NO_SCRATCH_RECLAIM=1` | `results-tuning/2026-08-03-6-acceptmap/` | 9/11 rc=0 with 36 rows, `#wrong`=0. **Acceptance is size-invariant** — same 3 of 6 pairs honoured at all 18 sizes, refuting the hypothesis that it varies by size. NVLS and PAT **silently substitute** → RING/SIMPLE; COLLNET_DIRECT and COLLNET_CHAIN **hard-error** (exit 3, `invalid usage`). WarpSpeed applies with LL as well as SIMPLE, never with TREE. |
| 2026-08-03T13:09:49Z | 23 (2 runs) | 14059 | amd-mi355x-4 | 1 | same, nothing forced | same logging config | `results-tuning/2026-08-03-6-acceptmap/REF_rep{2,3}.*` | Rerun of the two repeats lost to the 14048 timeout. With rep1: 3/3 rc=0, **0 selection mismatches**, and **0 differences vs the old VERSION-logged baseline** — the logging change did not perturb selection. Verified channel counts 1→224. This run supersedes batch 3 as the comparable baseline. |

**Run loss, my error:** job 14048 hit its 4-hour limit at 13:08:14Z (`sacct`: `TIMEOUT`, exit 0:0 —
not `NODE_FAIL`). I allocated `-t 240` at 09:08 and did not track expiry. `REF_rep2` was SIGTERMed
mid-run and `REF_rep3` never started; both rerun on job 14059 on the **same node**, so all three
repeats remain comparable.

## Batch 7 — acceptance map across 5 more collectives

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-03T13:2xZ | ~60 (5 runs) | 14059 | amd-mi355x-4 | 1 | Phase 0 smoke: `<coll>_perf -b 256M -e 512M -f 2 -g 1 -n 5 -w 2 -c 1 -A 1` | `NCCL_DEBUG=VERSION` | `results-tuning/2026-08-03-7-collmap/smoke_*.log` | `-A 1` populates columns for all_gather, reduce_scatter, broadcast, reduce — but returns **`N/A`** for alltoall (no `getAlgoProtoChannels` callback). broadcast/reduce show 112 channels, not 56. |
| 2026-08-03T13:3xZ | ~450 (37 runs) | 14059 | amd-mi355x-4 | 1 | `<coll>_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1 -Z csv -X <file>` via `srun --mpi=pmix --ntasks-per-node=8` | 5 collectives × (RING,TREE × SIMPLE,LL128,LL) + PAT probe on all_gather/reduce_scatter + 1 unforced ref each. `NCCL_DEBUG=INFO` + `NCCL_DEBUG_FILE=..._%p.log`, `LD_PRELOAD=$MY_PATH/librccl.so`, `HSA_NO_SCRATCH_RECLAIM=1` | `results-tuning/2026-08-03-7-collmap/` (37 CSVs, 42 logs, `SUMMARY.md`) | 37/37 rc=0. **TREE honoured only for all_reduce.** **LL128 never honoured by any collective.** **PAT never honoured** — contradicts `tuning.cc`. **all_gather is unsteerable below 16M** (`Direct` path, forcing has no effect); only collective where acceptance is size-dependent. alltoall has no algo/proto concept. broadcast ≡ reduce, channel ceiling 112 not 56. Of 36 sweepable combos only ~10 distinct real configs exist. |

Also established: `-A 1` and INFO report **different layers** — base algo and proto agree 18/18,
but `-A 1` additionally exposes the RCCL addon (`RING*` = WarpSpeed, `Direct`) that INFO's
per-call line omits. `unsupported_combos.yaml`'s six existing rules are all correct; it is missing
LL128, PAT, and the all_gather-below-16M case.

## Batch 8 — multi-node wireup diagnosis (7 smoke attempts)

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-03T15:18:25Z | ~600 | 14067 | amd-mi355x-3,5 | 2 | `all_reduce_perf -b 256M -e 512M -f 2 -g 1 -n 5 -w 2 -c 1 -A 1` via `srun --mpi=pmix`, 7 variants | progressively: bare, `OMPI_MCA_btl` variants, port ranges, mpirun-over-ssh, finally all env inside `bash -c` | `results-tuning/2026-08-03-9-multinode/smoke_*.log` | 6 failed, 7th passed. **Root cause: `srun --export` treats commas as its own separator**, so `OMPI_MCA_btl=self,vader,tcp` became `btl=self` and `NCCL_IB_HCA=ionic_0:1,...` became one HCA. Fix: export inside a `bash -c` wrapper. Multi-node also requires the fabric env + `OMPI_MCA_btl=self,vader,tcp` + `OMPI_MCA_btl_tcp_if_include=enp81s0f1np1`. |

## Batch 9 — multi-node defaults + accepted combinations (2 and 3 nodes)

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-03T15:30Z | ~1500 (58 runs) | 14067 | amd-mi355x-3,5 | 2 | `<coll>_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1 -Z csv -X <file>` via `run_mn.sh` | 6 colls × (RING,TREE × SIMPLE,LL128,LL) + 3 unforced repeats each + 4 exotic probes. Fabric env + `NCCL_DEBUG=INFO` → per-rank file, all inside `bash -c` | `results-tuning/2026-08-03-9-multinode/2n/` | 57/58 rc=0. **All 6 all_reduce algo/proto pairs honoured at 2 nodes** (vs 3 at 1 node) — LL128 and TREE/SIMPLE both work. **48-channel cap CONFIRMED**: `RCCL MaxChannels: default capping to 48` fires 14,400× (0× at 1 node), real channels 47–48. Multi-node defaults are LL128-dominated and structurally unlike single-node. No `RING*` anywhere. |
| 2026-08-03T16:00Z | ~1600 (58 runs) | 14067 | amd-mi355x-3,5,6 | 3 | as above | as above | `results-tuning/2026-08-03-9-multinode/3n/` | 54/58 rc=0 (4 alltoall failures, rc=137). Near-identical to 2 nodes; differences: all_gather `Direct` uses 64 channels (not 32, and **not** subject to the 48 cap), all_reduce holds RING/LL128 to 256M, reduce_scatter mid-range channels differ. Unforced selection identical across 3 repeats for every collective. |

Both scales used a subset of one allocation, so 2n and 3n are directly comparable.
alltoall is **intermittently unstable multi-node** — 5 failures total, retries sometimes pass.
Proposed `findings/` update is drafted at the end of
`results-tuning/2026-08-03-9-multinode/SUMMARY.md`, **awaiting approval**.

## Batch 10 — fixed-tool sweep, attempt 1 at `-n 5` (SUPERSEDED, data discarded)

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-04T12:09:10Z | ~960 (93 runs) | 14138 | amd-mi355x-7 | 1 | `rccl_sweep.py` → `all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 5 -w 5 -c 1 -A 1 -C 1`; 3 combos × 10 channel values × 3 repeats + default | `NCCL_DEBUG=VERSION` (sweep_config.yaml); `MY_PATH=/opt/shared/ylerman/GPU-107/bin`; launched by the tool via mpirun-over-ssh | `results-tuning/2026-08-04-1-sweep1node/superseded-n5/` | **Data discarded.** 93/93 rc=0, **0 substituted** (so the tool fixes worked). But `-n 5` gave a **median 10–20% repeat spread, worst 193%** — larger than `optimize_metrics.py`'s 5% equivalence tolerance, so its min-channels tie-break selected 1-channel configs at 16M–64M. `generated_n5.conf` is visibly broken. Root cause: `sweep_config.yaml` had `iterations: 5`. |

## Batch 11 — fixed-tool sweep at `-n 20` (in progress)

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-04T12:29Z | ~1400 (93 runs) | 14138 | amd-mi355x-7 | 1 | `rccl_sweep.py` → `all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1 -C 1`; same matrix | as above, `iterations: 20` | `results-tuning/2026-08-04-1-sweep1node/` | In progress. rep1 complete (31 runs, 0 substituted). |

Sweep matrix both batches: all_reduce, 1 node, channels {1,2,4,8,16,24,32,40,48,56} ×
{RING/SIMPLE, RING/LL, TREE/LL} × 3 repeats, + 1 unforced default per repeat = 31 runs/repeat.
Only the 3 algo/proto combinations verified as honoured at 1 node were swept.

Exact per-stage command chain: `results-tuning/COMMANDS.md`.
`-n`/`-w`/`-c`/`-A`/`-C` come from `sweep_config.yaml`, not the CLI — see that file.

Tool fixes active from batch 10 onward (`tools/rccl-sweep/`): `-A 1` not `-M 1`; `-C` not `-R`;
measured-not-requested selection recorded with a `substituted` flag; 6 audit columns in the DB and
CSV; `optimize_metrics.py` drops substituted rows; `unsupported_combos.yaml` carries the measured
honoured-combination map.

## Batch 12 — 1-node A/B: generated config vs RCCL default

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-04T13:36Z | ~330 (14 runs) | 14138 | amd-mi355x-7 | 1 | `validate_tuner_config.py --repeats 7 --plugin-must-fire`, each run `all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1` via `srun --jobid=14138 --ntasks-per-node=8` | config arm: `NCCL_TUNER_PLUGIN=ab-tuner-test/librccl-tunerv4-dn.so` + `NCCL_TUNER_CONFIG_FILE=candidate.conf` (md5 `dc515bba`). Both arms `NCCL_DEBUG=INFO`→`NCCL_DEBUG_FILE`. | `results-tuning/2026-08-04-2-ab1node/` | **8 of 12 rules verified wins**, P(sup)=1.00 each: +12.5% @4K, +29.1% @8K, +45.3% @16K, +56.1% @32K, +60.2% @64K, +32.3% @128K, +39.5% @256K, +63.0% @512K–1M. 4 rules dropped, incl. **−65.7% @64M–512M**. 0/7 hangs both arms, `#wrong`=0 on all 252 rows. Whole-sweep avg busbw 143.89→90.37 (−37.2%) — the config *as written* is a net loss; the validated subset is the shippable output. |

**Cause of the −65.7%, corrected.** My first reading was "the config lost WarpSpeed". **Wrong** — `RING*`
and `WarpSpeed enabled` appear in both arms, all 8 ranks. The real cause is a **channel-unit mismatch
between the env-var and plugin paths**:

| path | request | base channels | actual (×4 WarpSpeed multiplier) |
|---|---|---|---|
| `NCCL_MIN/MAX_NCHANNELS=48` (what the sweep measured) | 48 | 48 | — |
| plugin `channels=48` (what was deployed) | 48 | **12** | 48 |
| default (unforced) | — | 56 | **224** |

`-A 1` reports 56 for default and **12** for the config at 64M–512M; the plugin's own log confirms it
applied `channels=48`. 48 ÷ 4 = 12, so the plugin's channel field is consumed as the *post-multiplier
total*, not the base. We therefore deployed ~4.7× fewer channels than default without intending to.

This is precisely the "measurement path != deployment path" failure `validate_tuner_config.py`'s
docstring warns about, now measured here. **Consequence: channel values taken from an env-var sweep
must not be written into a plugin config unchanged wherever WarpSpeed is active (>=64M for
all_reduce).** Below 64M no multiplier applies and the two paths agree — which is why all 8 wins are
at 4K-1M and every rule >=2M either drew or lost.

## Batch 13 — 2-node sweep (config creation)

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-04T13:43:01Z | 1889 (165 runs) | 14138 | amd-mi355x-5,7 | 2 | `rccl_sweep.py` → `all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1`; all 6 algo/proto combos × channels {1,2,4,8,16,24,32,40,48} × 3 repeats + default | fabric env from `sweep_config.yaml`; `MY_PATH=…/bin`; mpirun-over-ssh launcher | `results-tuning/2026-08-04-4-sweep2node/` | 165/165 rc=0, **0 substituted of 2970 rows** (all 6 combos honoured, as findings/02 predicted). **0 `RING*` rows** — WarpSpeed never engages at 2 nodes. busbw spread median 0.8%, worst 80%, 72/990 >5%. Channel list capped at 48 per the measured multi-node cap. |

## Batch 14 — 2-node A/B

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-04T14:16:35Z | 191 (14 runs) | 14138 | amd-mi355x-5,7 | 2 | `validate_tuner_config.py --repeats 7 --plugin-must-fire`, `-n 20 -w 5 -c 1 -A 1` via `srun --jobid=14138 --ntasks-per-node=8` | config arm: plugin + `candidate.conf` md5 `bfaeecc9`. Both arms `NCCL_DEBUG=INFO`→file + fabric env. | `results-tuning/2026-08-04-3-ab2node/` | **1 of 13 rules kept**: `32M ring/ll128 40ch` **+11.1%** P(sup)=1.00 (an *algorithm* switch — default picks TREE there). Everything else drew or lost ≤3.9%. 7 of 18 sizes had a config identical to default. No landmine — WarpSpeed absent, so the 1-node channel-unit trap cannot occur. 0/7 hangs. |

## Batch 15 — 3-node sweep (config creation)

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-04T14:20:31Z | ~2200 (165 runs) | 14138 | amd-mi355x-5,6,7 | 3 | as batch 13 but `--nodes 3` | as batch 13 | `results-tuning/2026-08-04-6-sweep3node/` | 165/165 rc=0, **0 substituted of 2970**, **0 `RING*`**. busbw spread median 0.7%, worst 66.7%, 78/990 >5%. |

## Batch 16 — 3-node A/B

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-04T15:01Z | ~190 (14 runs) | 14138 | amd-mi355x-5,6,7 | 3 | as batch 14, `--nodelist amd-mi355x-5,6,7` | config arm: plugin + `candidate.conf` md5 `33f167f9`. Plugin firing verified in all 24 per-rank logs. | `results-tuning/2026-08-04-5-ab3node/` | **2 of 13 rules kept**: `32M ring/ll128 32ch` +8.7%, `64M–128M ring/ll128 40ch` +10.2%, both P(sup)=1.00. **RCCL's 3-node default has a real dip at 128M** (247.70 vs 268.78 at 64M — bandwidth falling as size rises); the config's 40 channels fixes it. Reproduces batch 9's independent observation. |

**Two genuine wins lost to rule granularity, not measurement** (3 nodes): `262144-1048576` contained
+8.9% at 256K *and* −5.0% at 512K; `268435456-536870912` contained +12.1% at 256M *and* −2.5% at
512M. Both correctly rejected as landmines. The ranges are only that wide because
`generate_tuner_config.py` merges adjacent sizes sharing identical settings. Per-size rules would
have kept **4** rules instead of 2. Tool limitation, not a hardware result — fix is to emit per-size
rules and let the validator prune, or let the validator split a mixed range.

## Scale trend across batches 12/14/16

| scale | kept | gains |
|---|---|---|
| 1 node | 8 of 12 | +12.5% … +63.0% |
| 2 nodes | 1 of 13 | +11.1% |
| 3 nodes | 2 of 13 | +8.7%, +10.2% |

RCCL's **single-node** defaults are loose (1–16 channels at small sizes, large headroom); its
**multi-node** defaults are tight (already at the 48-channel cap). Multi-node wins come from
algorithm choice or from correcting a specific default misstep, not from retuning channels broadly.

## Batch 17 — combined validated config, end-to-end verification

| utc_start | duration_s | jobid | nodelist | n_nodes | command | non-default env | output path | result |
|---|---|---|---|---|---|---|---|---|
| 2026-08-04T15:4xZ | ~15 (1 run) | 14138 | amd-mi355x-7 | 1 | `all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1` via `srun --jobid=14138 --ntasks-per-node=8` | `NCCL_TUNER_PLUGIN` + `NCCL_TUNER_CONFIG_FILE=validated_all_scales.conf` (md5 `d34745da`), `NCCL_DEBUG=INFO`→file | `/opt/shared/ylerman/GPU-107/verifycombined-2026-08-04/` and `results-tuning/2026-08-04-7-validated/` | **Merge verified.** 8 one-node rules fire at 4K–1M; 2M–512M falls through to default (`RING/SIMPLE/56`, `RING*/SIMPLE/56` from 64M); 2-node and 3-node rules correctly do NOT fire, so `nNodes`/`nRanks` scoping works. ≥64M keeps `RING*`/56 — WarpSpeed intact, landmine avoided by omission. |

**Deliverable**: `results-tuning/2026-08-04-7-validated/validated_all_scales.conf` — 11 rules, all
verified at P(sup)=1.00. See that folder's `README.md` for deployment, gains, scope and the
plugin-channel-semantics warning.

Header-line quirk recorded there too: `dn/dn-tuner/plugin.c:199` parses the CSV header as an extra
harmless rule (hence "Allocated memory for N+1 configurations"), while
`validate_tuner_config.py`'s `parse_rules()` requires it. The file keeps the header — correct for
the validator, harmless to the plugin.

## Corrections to batch 16 (3-node A/B), 2026-08-04

Two independent subagents built the 3-node page and both audited my `SUMMARY.md` against the raw
logs. Their findings, and what changed:

| my claim | reality |
|---|---|
| "dip: 269.30 → 248.51" | those were **repeat 1**, not medians. Medians of 7: **268.78 → 247.70** (−7.8%). |
| "firing verified in all 24 per-rank logs" | **true, but the local evidence was incomplete** — only 7 of 168 config per-rank logs had been copied. Cluster has 168/168, all containing `Applied config`; 0/168 default logs mention the plugin. All 168 now copied locally. |
| "32M win is an algorithm switch" | it is **algo *and* channels** (TREE/LL128/48 → RING/LL128/32). |
| "several rules are exact +0.0%" | exactly **four** (4K, 8K, 16K, 64K). 8M is +0.05% and only *prints* as +0.0%. |
| the 4M outlier | **unreported by me**: default repeat 6 measured 11.69 GB/s against a 61.91 median (82% spread), the largest in the run. Does not change the verdict — P(sup) is a rank statistic — but the run must not be read off means. |
| "config's 40 channels: no dip" | true **in-place**; out-of-place the config's 128M r3 overlaps its 64M range. The *default's* dip holds in both placements with zero overlap, so that is the robust half. |

Strengthened rather than weakened: the dip is non-overlapping across all 49 repeat pairings in both
placements, and the **mechanism** is visible in the selection logs — the 3-node default matches the
2-node default at 16 of 18 sizes, and the only two that differ (128M, 256M) are exactly the two
where this config wins. 3 nodes holds `RING/LL128` where 2 nodes has already moved to
`RING/SIMPLE`.

**Process note.** I dispatched a second agent for this page after `TaskList` showed no tasks,
concluding the first was lost in the session restart. `TaskList` does not track background Agents,
so the first was still alive: both wrote the same path and one clobbered the other. Disk holds the
second agent's 93 KB version (md5 `7cbcdc9e`); the first agent's 97 KB version is preserved at
`/tmp/gpu107-3n-ylerman/presentation.mine.html`. Both are structurally valid with zero external
references. Lesson: confirm a background agent is actually dead before re-dispatching to the same
output path.

| 2026-08-17T07:41Z | 1440 (69 runs: 66 search + 3 default) | 14470 (peer's alloc, agreed quiet window 07:39-08:05Z; own job 14570 stayed PD) | amd-mi355x-7 | 1 | `adaptive_search.py live --nodes 1 --grid 1,2,4,8,16,24,32,40,48,56 --anchors 8,24,48 --margin 15 --repeat-policy 3 --tol 0.5` → per-run `rccl_sweep.py` → `all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1`, one config per invocation, 3 repeats + 3 default runs | sweep_config.yaml env; `MY_PATH=/opt/shared/ylerman/GPU-107/bin`; mpirun-over-ssh from node 7 | `results-tuning/2026-08-17-1-optuna1node/` (remote: `/opt/shared/ylerman/GPU-107/optuna-live-2026-08-17/1n`) | **Adaptive policy validated out-of-sample: 69 runs vs the grid's 93 (-26%), 69/69 rc=0, 0 substituted, 22 configs.** vs 2026-08-04 winners (tol 0.5): 12/18 exact; all 6 diffs are channel-only and at each the live pick measured >= the 08-04 winner config in the same session (worst live-side gap 0.0%). raw runs in remote/ (69 command.txt, 552 dbg logs). compare_vs_0804.json |
