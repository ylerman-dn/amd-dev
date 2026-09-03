# stackcmp — decompose the old-site vs container gap at 4K-32M (2026-09-03, des2-2)

Question: old site 1-node numbers (e.g. 1M default = 41.46 GB/s) vs today's container
bracket-sweep numbers (1M = 24.84 at 112ch) differ ~2x. Which factor: launch mode,
library, container, MSCCL?

4 cells x 5 interleaved reps, `all_reduce_perf -b 4K -e 32M -f 2 -n 20 -w 5 -c 1 -A 1`:
- **a_mpi_fork**: bare metal, fork librccl (LD_LIBRARY_PATH+LD_PRELOAD per
  `tools/rccl-sweep/sweep_executor.py:163-165`), 8 MPI ranks x `-g 1` via srun — old-site replica.
- **b_sp_fork**: same fork, 1 process x `-g 8`.
- **c_ct_m0**: container stock RCCL, `-g 8`, `RCCL_MSCCL_ENABLE=0` — bracket-sweep replica.
- **d_ct_mon**: container stock RCCL, `-g 8`, MSCCL enabled (image default = SGLang deployment env).

## Verdicts (medians; full table + raw reps in analysis.txt)
1. **Old site numbers REPRODUCE** (a_mpi_fork within 2-8% of the site rows at all 14 sizes,
   on a different node). No drift; the old table is valid on its own stack.
2. **Launch mode is the dominant factor**: same fork, same bare metal, 1M drops 40.00 -> 20.75
   (-48%) going from 8 MPI processes to one `-g 8` process; similar -35..-55% at 4K-4M,
   fading by 16M. The container sweeps measure the slower single-process regime.
3. **Library/container is minor and favors stock**: c_ct_m0 vs b_sp_fork = +5..+17% at most sizes.
4. **MSCCL nearly closes the gap**: d_ct_mon ~= old site at 4K-2M (1M: 38.92 vs 41.46) and
   BEATS the old site at 4M (142.30 vs 128.23); crossover ~8-16M, above which MSCCL-on is
   slightly worse (16M: 242.97 vs 255.09) and irrelevant by 32M.

## Implications (hypotheses for follow-up, not findings)
- The bracket sweep's "112 wins everywhere" holds within the MSCCL-off, single-process regime
  only. In the deployment env (SGLang image, MSCCL on by default) small/mid allreduces are
  MSCCL-served (2026-08-30 root cause: mscclFuncAllReduce, tuner never consulted) and run
  ~40-60% faster than the tuner-visible path at 128K-4M in this benchmark.
- All in-model A/Bs to date forced `RCCL_MSCCL_ENABLE=0` in BOTH arms; deployment default
  (MSCCL on) was never A/B'd in-model. Decode-size busbw here suggests MSCCL-on could beat
  both arms at those sizes — needs an in-model MSCCL on/off A/B before any conf ships.
- `-A` selections in d_ct_mon still print TREE/LL/RING/SIMPLE/112 while busbw doubles —
  consistent with `-A` reporting the RCCL planner's choice even when MSCCL executes; do not
  use `-A` to detect MSCCL.

## Ops
- des2-2 (TEST), allocs 20987 + 20989 (A/B rerun), both released. First A/B attempt segfaulted:
  LD_PRELOAD alone mixes fork `librccl.so` with host `/opt/rocm/lib/librccl.so.1`
  (`rcclGetAlgoInfo` backtrace); fixed by adding the tool's `LD_LIBRARY_PATH`. Raw logs:
  des2-2 `/data/ylerman/stackcmp-2026-09-03/` (stdout + per-rank INFO + progress.log).
