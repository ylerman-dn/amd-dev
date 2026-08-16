# 0.5% tolerance config, A/B'd at 2 and 3 nodes — 2026-08-16

Job 14467, nodes 5,6,7 (same sets as 2026-08-04 and as the 1% run). 7 repeats per arm,
8 warm-up runs, preflight 5, `--split-ranges`, plugin firing verified.

## 3 nodes: 4 of 12 rules kept (the 1% config kept 3 of 11)

| size | 1% config | 0.5% config |
|---|---|---|
| 256K | +12.7% | +11.7% |
| 32M | dropped, P(sup) 0.86 | **+7.2%, P(sup) 1.00** |
| 64M–128M | +12.4% | +11.8% |
| 256M–512M | +16.7% | **+17.3%** |

**32M converted.** With the 1% config it measured +7.6% but one stalled repeat in the config arm
(12.33 against 191–202 for the other six) dragged P(sup) to 0.86. Here it reproduces cleanly. That
confirms the earlier drop was a measurement artefact rather than a real failure — the gain was
always there.

## 2 nodes: 3 of 13 kept, statistically the same as 1%

| size | 1% config | 0.5% config |
|---|---|---|
| 256K–512K | +7.1% | +6.6% |
| 32M | +10.8% | +9.5% |
| 128M(–256M) | +2.7% | +3.3% |

Differences are within run-to-run variation. No evidence either tolerance is better at 2 nodes.

## Validity

- preflight passed both scales: 14.3% worst spread at 2n, ok at 3n
- 0 hangs in 28 measured runs
- within-arm spread: 1 of 36 points over 25% at 2n, 2 of 36 at 3n — both far below the 20%-of-points
  invalid threshold

## Two rules the 0.5% config lost that are worth noting

- `8192 ch4 3n` dropped as a landmine at **−4.5%**. The 1% config saw +0.0% at that size. 8K is
  latency-bound and tiny (0.21 GB/s), so this is likely one disturbed repeat rather than a real
  regression — but it is correctly refused either way.
- `1048576-16777216 ch48 3n` dropped for a −4.3% inside the range. Same rule was −0.3%/−1.6% under
  the 1% config.

## Conclusion

At 3 nodes 0.5% is the better setting: one more verified rule and a larger top-end gain. At 2 nodes
the two are indistinguishable. Nothing here argues for going back to 1%, and both are far better
than the original 5%.
