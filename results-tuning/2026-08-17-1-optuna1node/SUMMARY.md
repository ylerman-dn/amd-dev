# 2026-08-17-1 — live 1-node adaptive sweep (out-of-sample test of the replay-frozen policy)

**What ran.** `adaptive_search.py live` on node 7 (inside job 14470's agreed quiet window,
07:39–08:05Z, ~24 min wall): the replay-frozen policy (anchors 8,24,48 · margin 15 ·
3 repeats · tol 0.5) driving one `rccl_sweep.py` invocation per run —
`all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1`, standard INFO-to-file logging.
Exact commands: `remote/r*/run_*/outputs/*/command.txt` (66 search) and `remote/d*` (3 default).

**Run count (the deliverable metric).** 69 command.txt files vs the full grid's 93
(2026-08-04 driver.sh) = **26% fewer runs**; excluding the 3 defaults both sides: 66 vs 90 =
27%, exactly what replay predicted. 69/69 rc=0, 0 of 1242 size-rows substituted,
22 of 30 grid configs evaluated, all 3 honoured combos survived the prune (expected at 1 node).

**Same winners?** vs the 2026-08-04 grid winners recomputed at the current tol 0.5
(`compare_vs_0804.json`):

- **12/18 sizes exact match** across a 13-day gap.
- All 6 mismatches (4K, 16K, 32K, 64K, 128K, 512K) are **channel-count-only** differences
  inside the same algo/proto, in the flat/noisy small-size band.
- The search *did* evaluate the 08-04 winner config live at all 6 — and at each one the
  adaptive pick measured **equal or higher busbw in the same session** (live-side gap ≤ 0.0%).
  I.e. judged on same-day data, the adaptive winner is never worse than the grid winner;
  the label differences are day-to-day drift among near-ties, the same drift the
  2026-08-16-12-variance data shows at these sizes.
- In the 08-04 data, the live picks were within 0.3–2.6% of that day's best, except 16K
  (TREE/LL/56, 5.1% below the 08-04 winner in 08-04's data — but 0.8% *above* it in today's).

**Caveat** (from the gpu107-tuner-v5 session, 2026-08-16-13-v5parity on branch
gpu107-tuner-v5): at 1 node the second run of a back-to-back pair measures 3–6% low at
4K–32K. This batch is one continuous stream (as was the 08-04 grid), so it is internally
self-consistent, but small-size winner labels across days should be read with that noise
in mind.

**Verification.** Selection verified per project standard: `-A 1` columns parsed per row
(algo/proto honoured, substituted=0 throughout); NCCL_DEBUG=INFO per-rank logs kept for every
run (552 files under `remote/`, 8 per run, `dbg_%p.log`). Raw stdout in each run dir's
`output.log`.

**Conclusion so far.** The racing/prune search transfers out-of-sample at 1 node: same-or-better
winners at 26% fewer runs, ~24 min wall vs ~35 for the grid. 2n/3n live batches pending a
clean-fabric window (hai_test2 co-tenants until ~12:02Z, then the v5 session's A/B, then ours).
