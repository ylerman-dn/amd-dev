# 2026-08-17-3 — live 3-node adaptive sweep (nodes 5,6,7)

**Gate first.** 3 RCCL-default runs at 3 nodes vs the 2026-08-16 3n default baseline:
**PASS** — every size ≥1M within 4.8% (uniformly ~−2%; hai_test2 was on [4,9] and the
amd-nic-metrics-exporter was active on all nodes). The uniform small negative shift means
absolute busbw carries a ~2% caveat; winner selection is unaffected (both sides of every
comparison measured under the same conditions). Gate runs in `gate/`, doubling as this
session's default-context runs. The v5 session's same-morning evidence that [4,9] is
leaf-local-harmless and that the metrics exporter causes single-run dips is at
`tuner-v5-ab-2026-08-16/EXPORTER_LOG` (their branch); my median-of-3 gate damps such dips.

**What ran (10:26–11:01Z,** inside job 14470's "WINDOW OPEN 3N"**).**
`adaptive_search.py live --nodes 3`, frozen policy (anchors 8,24,48 · margin 15 ·
3 repeats · tol 0.5), channels 1–48, all 6 combos. Per-run command identical to the grid
sweep's (`-n 20 -w 5 -c 1 -A 1`, INFO-to-file).

**Run count.** **102 search runs vs the grid's 162 (105 vs 165 with defaults, −36%)** —
again exactly the count replay predicted. 102/102 rc=0, 0 substituted, 34 of 54 configs,
RING/LL pruned at the anchor stage (same as 2n).

**Same winners?** vs the 2026-08-16 3n grid, tol 0.5 (`compare_vs_0816.json`):

- **15/18 exact.**
- 3 mismatches (256K, 64M, 512M): all channel-count-only within the same algo/proto.
  At 256K and 64M the live pick measured *better* than the 08-16 winner config in the
  same session (−0.96%, −2.74% live-side gaps). At 512M the live pick (RING/SIMPLE/40)
  sat 0.47% below the 08-16 winner (48ch) in-session — inside the 0.5% tie band, i.e.
  the fewest-channels tie-break firing on a genuine tie.
- The known-noisy 64K–128K sizes (46–61% spread in the baseline itself) matched exactly.

**Verification.** `-A 1` per-row selection checks (0 substituted); per-rank INFO logs for
all runs under `remote/` (24 ranks × 102 runs) and `gate/`.

**Conclusion.** The adaptive policy holds at 3 nodes: −36% runs, 15/18 exact, all
misses are ties. Combined with 1n (−26%, 12/18 + 6 ties-or-better) and 2n (−36%, 14/18
+ 4 ties-or-better), the racing search is validated live at every scale it was designed
for.
