# broadcast/reduce 512K default dip is single-node-only; dip detection triages collectives

RCCL's default loses bandwidth as the message grows at 512K for broadcast (13.0 → 10.2
GB/s, −22%) and reduce (12.3 → 9.8, −20%) — at 1 node only. At 2 nodes both default
curves rise monotonically. The dip sits exactly where the project's largest verified
wins live (+148.3% / +143.5% @512K, forcing ring/ll/32). The hotspot detector
(`detect_hotspots.py`, threshold 10%) fires on exactly these two cases out of all six
collectives' 1-node default curves, and on nothing at 2 nodes — correct triage. At 3 nodes
broadcast has a distinct dip (−20.5% @64K, nodes {1,3,6}) that the detector also catches;
reduce 3n is clean. Its one false-negative is the 3-node all_reduce 128M dip (−8.9%,
under the 10% default trigger; caught at 5%).

**Evidence:** dip measured three times — `results-tuning/2026-08-03-7-collmap/*__REF.csv`,
the A/B in `2026-08-16-15-bcast/` (parent branch), and `2026-08-17-4-hotspot-mn/`
(1n vs 2n, 3 repeats, stable in every repeat) plus fresh 1n defaults in
`2026-08-18-1-newcolls/`. Reproduce: `detect_hotspots.py <curve.csv> --threshold 0.10`.

**Date:** 2026-08-17/18. **Scope:** RCCL 2.28.3-develop:2e42aa8, MI355X; 1/2/3 nodes
all measured (`2026-08-18-1-newcolls/hotspots_mn_th10.csv`).

**Falsified by:** a 2-node broadcast/reduce default curve showing a ≥10% dip, or the
1-node/3-node dips absent after an RCCL upgrade.
