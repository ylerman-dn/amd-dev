# 2026-08-17-2 — live 2-node adaptive sweep (nodes 5,7)

**Gate first.** hai_test2 was running on [4,9] during this batch, so the batch was gated:
3 RCCL-default runs at 2 nodes, compared to the 2026-08-16 2n default baseline
(`2026-08-16-12-variance/variance.json` default rows). **PASS** — every size ≥1M within
2.5% of baseline (worst −2.42% @128M), supporting the topology argument that {5,7}
(both L1-leaf) shares no fabric links with [4,9] (both L2-leaf). Gate runs kept in
`gate/`; they double as this session's default-context runs.

**What ran (09:02–09:32Z,** inside job 14470's agreed "WINDOW OPEN 2N"**).**
`adaptive_search.py live --nodes 2` with the same frozen policy (anchors 8,24,48 ·
margin 15 · 3 repeats · tol 0.5), channel grid 1–48, all 6 algo/proto combos in play.
Per-run command identical to the grid sweep's: `all_reduce_perf -b 4K -e 512M -f 2 -g 1
-n 20 -w 5 -c 1 -A 1`, INFO-to-file logging.

**Run count.** **102 runs vs the full grid's 165 (−38%)** — and exactly the 102 that the
replay of the 08-16 grid predicted. 102/102 rc=0, 0 substituted rows, 34 of 54 configs
evaluated. **RING/LL was pruned at the anchor stage** — the first live firing of the
combo prune (at 1 node all combos survive; at 2 nodes RING/LL loses to RING/LL128
everywhere).

**Same winners?** vs the 2026-08-16 grid (measured 1 day earlier on the same nodes),
tol 0.5 (`compare_vs_0816.json`):

- **14/18 exact.**
- 4 mismatches (32K, 64K, 128K, 256M): all channel-count-only, same algo/proto.
  At each, the 08-16 winner config was also measured live, and the live pick scored
  equal or better in the same session (live-side gaps 0.0 to −1.0%). Reference-side
  gaps ≤1.21%. The 256M case is a pure tie-break flip: in the 08-16 data itself the
  live pick (48ch) had *higher* bandwidth than that grid's winner (40ch) — 40 won there
  only via the fewest-channels rule inside the 0.5% band.

**Verification.** `-A 1` selection columns parsed per row (0 substituted); per-rank
`dbg_%p.log` INFO logs for all runs under `remote/` (16 ranks × 102 runs) plus `gate/`.

**Conclusion.** The adaptive policy transfers out-of-sample at 2 nodes: −38% runs,
winners same-or-better, and the combo-prune stage now proven live. 3-node batch pending
the v5 session's A/B completing (its allocation, agreed serialization) and [4,9] going
quiet for real spine-sharing reasons.
