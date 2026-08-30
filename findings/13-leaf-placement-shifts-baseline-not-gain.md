# 13 — Leaf placement shifts the baseline, not the tuning gain

**Claim.** Crossing the spine lowers the 2-node all_reduce baseline by up to 17% at small sizes
and by nothing above 8M, but a shipped tuning rule keeps its gain in both placements.

**Evidence.**

Same-leaf `amd-mi355x-3,amd-mi355x-5` (both L1) versus cross-leaf `amd-mi355x-3,amd-mi355x-9`
(L1 to L2, over the shared spine), node 3 anchored in both arms, 5 interleaved repeats per arm,
all 20 runs rc=0. Tuned runs verified by `Applied config` in 16/16 rank logs, default runs 0/16.

Baseline shift (default arm, median busbw, cross versus same):

| 4K | 16K | 32K | 128K | 512K | 1M | 8M | 32M | 512M |
|---|---|---|---|---|---|---|---|---|
| -9.2% | **-17.4%** | -13.3% | -8.8% | -7.0% | -2.9% | +0.8% | -0.2% | +0.2% |

Gain at 32M, the size where the shipped 2-node config has a rule — all five repeats:

| arm | repeats (GB/s) | median | gain |
|---|---|---|---|
| same-leaf default | 222.9 218.8 220.2 222.2 221.9 | 221.85 | - |
| same-leaf tuned | 240.1 243.0 243.2 247.6 240.2 | 242.95 | **+9.5%** |
| cross-leaf default | 222.1 220.0 217.9 221.4 224.1 | 221.41 | - |
| cross-leaf tuned | 238.0 237.4 239.8 243.7 238.4 | 238.41 | **+7.7%** |

Every tuned repeat beats every default repeat in both placements — no overlap.

Reproduce: `/opt/shared/ylerman/GPU-107/leaf-2026-08-30/run_leaf.sh <jobid>`
(copy in `results-tuning/2026-08-30-1-value/leaf-ab/run_leaf.sh`).

**Date.** 2026-08-30.

**Scope and limits.** 2 nodes, 16 ranks, all_reduce only, sizes 4K-512M,
RCCL 2.17.9-develop:2e42aa8. A co-tenant held nodes 1,4,6,7 across both leaf groups
throughout, so the spine was not quiet — that can only inflate the cross-leaf penalty, not
create it. The config's other rule, at 256K, could **not** be resolved: repeat spread there is
7-12% and a no-rule 16K control swung +6% on noise alone, so 5 repeats cannot see a 5% effect
at small sizes.

**Falsified by.** A cross-leaf run where the 32M tuned arm fails to beat the cross-leaf default.
