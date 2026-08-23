# How dip triage works — worked example, real numbers

## Where it sits

BEFORE any search. It is not a search method: it decides WHICH of the 18
sizes deserve searching at all. The searcher (adaptive or grid) then runs
only on the flagged sizes. The alternative is what we do by default: search
all 18 sizes.

## The physics

busbw must rise (or flatten) as message size grows - larger payloads amortize
fixed overheads. The default config's curve therefore may never DROP. A drop
is proof that RCCL's default is mistuned at that size, and the proof costs
one measurement (3 runs) of the default config.

## Real example - broadcast 3n (2026-08-18 data)

Default curve, the relevant part: ... 32K: 0.79, 64K: 0.70, 128K: 1.55 ...
running max before 64K is 0.88. Detector rule at 10%:
0.70 < 0.88 x 0.9 = 0.79 -> size 64K flagged (a 20.5% dip). No other size
violates the rule -> flag set = {64K}.

Then adaptive runs restricted to 64K: total cost 14 measurements
(1 default + 13 search) instead of 24 for adaptive-over-all-sizes.
The 64K fix it finds is the same one the grid finds.

## The blindness, same dataset

At 128K - the size right next to the dip - the default (1.55) obeys the
running-max rule, so 128K is NOT flagged. But the grid winner at 128K is 59%
faster than the default. Same story at 256K (51%) and 512K (28%). The default
there is uniformly mediocre: slow at every neighbouring size in similar
proportion, so the curve shape shows nothing. Measured across all 15 grids,
searching only triage-flagged sizes leaves up to 59% (broadcast 3n) and 58%
(reduce 2n) on the table.

Dips prove guilt; a smooth curve does not prove innocence.

## Where it fits

1. Tiny budgets: on a contended production cluster, triage points at the
   guaranteed wins for ~1 measurement.
2. Regression monitoring: after a full tune, re-measure the default curve
   periodically; a NEW dip = alarm (RCCL upgrade, fabric change). This is its
   best role.

What it must never be: the only tuning method.
