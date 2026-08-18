# 2026-08-17-4 — is the broadcast/reduce 512K dip multi-node too? (hotspot detector inputs)

**Question.** Halperin's hotspot detector fires on broadcast and reduce at 1 node
(−21.6%/−20.8% @512K on the 2026-08-03 collmap curves) — exactly where the +148%/+143%
verified wins live. Does the same dip exist at 2 nodes?

**What ran (11:32–11:36Z, own job 14583, nodes 6+8).** 15 default runs (no algo/proto/
channel forcing, standard `-n 20 -w 5 -c 1 -A 1`): broadcast ×3 and reduce ×3 at 1 node
(node 8) and at 2 nodes ({6,8}), plus all_reduce ×3 at 2 nodes as a control. NOTE: {6,8}
is an L2-leaf pair, not the published {5,7} — valid here because hotspot detection is
self-relative (a curve against its own running max). Control: the {6,8} all_reduce 2n
default medians land within ~2% of the {5,7} 2026-08-16 baseline (worst −5.1% @2M, caused
by one dipped repeat), so the pair/fabric are sound.

**Result: the dip is 1-node-only.**

- Detector at 10% threshold on all five fresh curves: **2 hotspots — broadcast 1n
  −21.9% @512K, reduce 1n −20.2% @512K.** Both stable in all 3 repeats (e.g. broadcast
  512K: 10.17/10.20/10.18 vs 13.04 at 256K) — real protocol-switch dips, not
  metrics-exporter noise. This also reproduces the collmap 1n finding on a different
  node (8 vs the collmap host).
- **2-node curves are monotonic: 0 hotspots, even at a 5% threshold.** Broadcast and
  reduce defaults rise with size the whole way at 2n.
- Single-run exporter-style dips did appear (reduce 1n 256K r2: 8.17 vs 12.3;
  reduce 2n 1M r1: 13.9 vs 16.8; all_reduce 2n 2M r2: 45.4) — all damped by the
  median-of-3, none created a false hotspot.

**Meaning.** The default's bad protocol switch at 512K afflicts broadcast/reduce only in
their single-node paths; at 2 nodes the defaults behave. So the hotspot loop's value on
this build is confined to 1-node broadcast/reduce (where it flags the two biggest wins
in the project) plus the 3n all_reduce 128M dip at a 5% threshold. Not checked: 3-node
broadcast/reduce (nodes were taken; same 15-minute protocol applies when the trio
frees up).

**Evidence.** `batch.log` (15/15 rc=0), `hotspot_mn_curves.csv` (median curves),
`hotspots_th10.csv` / `hotspots_th05.csv` (detector reports), raw runs under `remote/`
(15 command.txt + per-rank INFO logs). Reproduce:
`python3 tools/rccl-sweep/detect_hotspots.py hotspot_mn_curves.csv --threshold 0.10`.
