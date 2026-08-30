# Leaf placement A/B — does the tuning gain survive crossing the spine?

Run 2026-08-30, allocation 20856. `all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1`,
5 repeats per arm, arms interleaved A/B/A/B so drift hits both equally.
Raw: this directory; remote `/opt/shared/ylerman/GPU-107/leaf-2026-08-30`.

- **same-leaf** `amd-mi355x-3,amd-mi355x-5` — both on L1, traffic never leaves the leaf
- **cross-leaf** `amd-mi355x-3,amd-mi355x-9` — L1 to L2, crosses the shared spine
- node 3 is the fixed anchor in both arms
- tuned arm loads `all_reduce_2n.final.conf` (2 rules: 256K and 32M)

All 20 runs rc=0 with 36 CSV rows. Verification is unambiguous: every tuned run shows
`TUNER/Plugin: Applied config` in all 16 rank logs, every default run in zero.

Co-tenant caveat: job 20854 held nodes 1,4,6,7 throughout, spanning both leaf groups, so
the spine was not quiet. That can only inflate the cross-leaf penalty, not create it.

## Result 1 — placement moves the baseline, biggest at small sizes

Median default busbw (GB/s), cross-leaf versus same-leaf:

| size | same-leaf | cross-leaf | delta |
|---|---|---|---|
| 4K | 0.18 | 0.16 | -9.2% |
| 16K | 0.76 | 0.63 | **-17.4%** |
| 32K | 1.45 | 1.25 | -13.3% |
| 128K | 4.83 | 4.40 | -8.8% |
| 512K | 16.25 | 15.11 | -7.0% |
| 1M | 27.99 | 27.19 | -2.9% |
| 4M | 89.65 | 87.22 | -2.7% |
| 8M | 134.50 | 135.60 | +0.8% |
| 32M | 221.85 | 221.41 | -0.2% |
| 512M | 378.94 | 379.55 | +0.2% |

The penalty decays monotonically with size and is gone by 8M — the signature of fixed
per-hop latency dominating small messages and disappearing once bandwidth-bound. This is a
live, controlled confirmation of what the offline comparison of stored runs suggested.

It also confirms, indirectly, that the node-to-leaf map in `CLAUDE.md:104-108` is right:
`{3,5}` and `{3,9}` behave exactly as same-leaf and cross-leaf should.

## Result 2 — the 32M rule's gain survives the spine

The shipped 2-node config has rules at 256K and 32M only. At 32M:

| arm | 5 repeats (GB/s) | median | gain |
|---|---|---|---|
| same-leaf default | 222.93 218.75 220.15 222.23 221.85 | 221.85 | - |
| same-leaf tuned | 240.08 242.95 243.24 247.59 240.24 | 242.95 | **+9.5%** |
| cross-leaf default | 222.12 219.95 217.86 221.41 224.09 | 221.41 | - |
| cross-leaf tuned | 237.99 237.39 239.78 243.71 238.41 | 238.41 | **+7.7%** |

Every tuned repeat beats every default repeat, in both placements — complete separation,
no overlap. The rule is topology-portable: it delivers on the same leaf and across the
spine, at roughly 80% of its same-leaf magnitude when crossing.

This also independently replicates the shipped 32M verdict (+11%, measured on different
nodes in `2026-08-24-1-stage2`).

## Result 3 — the 256K rule cannot be resolved at 5 repeats

| arm | 5 repeats (GB/s) | median | spread |
|---|---|---|---|
| same-leaf default | 8.34 8.53 7.74 8.50 8.64 | 8.50 | 11.6% |
| same-leaf tuned | 8.97 8.20 8.98 8.96 8.73 | 8.96 | 9.6% |
| cross-leaf default | 7.78 7.42 7.39 7.90 7.85 | 7.78 | 6.9% |
| cross-leaf tuned | 8.40 8.41 8.06 8.04 7.74 | 8.06 | 8.6% |

Medians say +5.5% same-leaf and +3.7% cross-leaf, but the distributions overlap heavily and
the repeat spread is 7-12%. A 16K control size, which has no rule and where the tuned arm
should be identical to default, shows spreads of 4-13% and apparent swings of +6% — pure
noise of the same magnitude as the claimed effect.

**At small sizes, 5 repeats cannot resolve a 5% effect on this hardware.** The 256K rule is
neither confirmed nor refuted here. The production A/B uses 9 repeats with a P(sup) gate
for exactly this reason.

## Answer to the question that prompted the run

Rules are topology-portable where they are measurable at all. The 32M gain survives the
spine with no overlap between arms. The tuner does not need a topology input on this
evidence. What placement changes is the *baseline*, by up to 17% at small sizes — which
means cross-scale comparisons between runs on different node sets carry a placement
component, and `RUNLOG.md` nodelists are the only record of it.

## What would falsify this

A cross-leaf run where the 32M tuned arm fails to beat the cross-leaf default, or a
same-leaf/cross-leaf pair at large size showing a baseline gap above the ~1% repeat spread.
