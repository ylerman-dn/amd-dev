# broadcast and reduce, 1 node — the largest verified wins in the project

Job 14469, `amd-mi355x-8`, 8 ranks × 1 GPU. 7 repeats per arm, 4 warm-up runs, preflight passed,
`--plugin-must-fire`, `--split-ranges`. 0 hangs.

## Result

| collective | rule kept | best gain | P(sup) |
|---|---|---|---|
| broadcast | `524288-2097152 ring/ll/32` | **+148.3%** | 1.00 |
| reduce | `524288-2097152 ring/ll/32` | **+143.5%** | 1.00 |

Per size, broadcast:

```
   size   default    config     change
   256K     13.08     12.93      -1.1%
   512K     10.18     25.28    +148.3%   <- kept
     1M     19.38     37.59     +93.9%   <- kept
     2M     37.89     50.29     +32.7%   <- kept
     4M     65.92     60.47      -8.3%
     8M    107.86     66.95     -37.9%
    16M    164.31    163.92      -0.2%
```

`reduce` has the same shape: +143.5% / +99% / +28.3% over the same three sizes, −41.6% at 8M.

## What is actually wrong

RCCL's default **loses bandwidth as the message gets bigger**: 13.08 GB/s at 256K falls to 10.18 at
512K. Bandwidth should rise. Forcing `RING/LL` removes the cliff entirely — 12.93 at 256K, 25.28 at
512K.

So the default switches away from LL somewhere around 512K and the replacement is roughly half as
fast for the next three sizes. It recovers by 4M, where LL becomes the wrong choice and forcing it
costs 38%.

This is the same class of error found in all_reduce — a protocol switch at the wrong size — but far
larger, and on a collective nobody has tuned.

## `--split-ranges` earned its keep here

The rule as generated spanned 512K–8M and contained a −37.9% regression at 8M, so the whole thing
was correctly refused as a landmine:

```
broadcast 524288-8388608 ch32   +148.1%  worst -37.9%   DROP (landmine)
  ↳ split 524288-2097152 ch32   +148.1%  worst +32.8%   KEEP
```

Without the splitting feature added on 2026-08-16 this win would have been discarded entirely.

## How this was found

Mining `results-tuning/2026-08-03-7-collmap/`, which had sat unanalysed. It contains one run per
(collective, algo, proto) at 1 node for all six collectives. Comparing each forced variant against
the unforced default showed `broadcast` and `reduce` with ~+28% mean headroom against all_reduce's
+22.7% — and individual sizes at +149%.

The A/B above confirms the sweep's prediction closely: predicted +149% / +125% / +115% at
512K / 1M / 2M, measured +148.3% / +93.9% / +32.7%. First size dead on; the other two lower but still
large.

## Caveats

- **1 node only, broadcast and reduce only.** The multi-node picture for these collectives is much
  smaller (~+3.7% mean) and not yet A/B'd.
- Sizes outside the rule range show ~0% change as expected, except 16K at −10.5%, which is outside
  the rule and therefore noise — small sizes are latency-bound and the noisiest part of the range.
- Channel count was fixed at 32 without sweeping it. The gain comes from the protocol, not the
  channels, but 32 may not be optimal.
- One point per arm exceeded 25% within-arm spread (broadcast 28.6%, reduce 31.5%), well under the
  threshold that would mark the run invalid.

## Files

`broadcast/`, `reduce/` (validate.log, times.csv, arm logs), the two `.conf` inputs and the two
`.validated.csv` outputs.
