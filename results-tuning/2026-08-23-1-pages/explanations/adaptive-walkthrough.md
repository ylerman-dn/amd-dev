# How the adaptive (racing) search works — worked example

Setting: broadcast at 3 nodes. Search space = 3 combos (RING/LL, RING/LL128,
RING/SIMPLE) x 9 channel counts {1,2,4,8,16,24,32,40,48} = 27 configs.
One measurement = median of 3 benchmark runs; each run reports all 18 sizes at
once, so every measurement carries the full size axis for free.
Policy: anchors = channels {1,8,24,48}, prune margin 15%, tie tolerance 0.5%,
hill-climb hysteresis 2%. (Numbers below are illustrative; the mechanism and
the policy values are the real ones from `adaptive_search.py`.)

## Stage 1 — anchors

Measure every combo at only the 4 anchor channels: 3 x 4 = 12 measurements.
Example medians at two of the 18 sizes:

busbw at 64K:

| combo        | ch1 | ch8 | ch24 | ch48 |
|--------------|-----|-----|------|------|
| RING/LL      | 1.2 | 2.0 | 1.8  | 1.5  |
| RING/LL128   | 2.6 | 2.2 | 1.9  | 1.6  |
| RING/SIMPLE  | 0.9 | 1.4 | 1.3  | 1.1  |

busbw at 64M:

| combo        | ch1 | ch8 | ch24 | ch48 |
|--------------|-----|-----|------|------|
| RING/LL      | 30  | 90  | 120  | 125  |
| RING/LL128   | 45  | 140 | 170  | 172  |
| RING/SIMPLE  | 40  | 130 | 165  | 168  |

## Stage 2 — prune combos dominated everywhere

Kill threshold per size = leader x 0.85.
- 64K: leader 2.6, threshold 2.21. LL best 2.0 (behind), SIMPLE best 1.4 (behind).
- 64M: leader 172, threshold 146.2. LL best 125 (behind), SIMPLE best 168 (alive).

A combo dies only if it is behind the threshold at EVERY size. If LL is behind
everywhere, all 9 of its channel configs are eliminated after only 4
measurements. SIMPLE survives on the strength of 64M alone.

## Stage 3 — per-size refinement, contenders only

At 64K: contenders (within 15% of leader at that size) = LL128 only. Its best
anchor is ch1 = 2.6. Hill-climb the neighbour: ch2 comes back 2.5; a new point
must beat the incumbent by more than 2% (hysteresis, a noise guard), so the
climb stops. Winner at 64K: RING/LL128/1.

At 64M: contenders = LL128 (172) and SIMPLE (168). LL128's best anchor is
ch48 = 172. Neighbour ch40 = 171.5 does not beat it by 2%, so ch48 stays the
incumbent. Then the downward tie-break walk: the 0.5% tie band is
172 x 0.995 = 171.14; ch40 = 171.5 is inside the band, so it is a tie and ties
go to the fewest channels -> step down again; ch32 = 165 falls outside the
band -> stop. Winner at 64M: RING/LL128/40.

## The bill

12 anchor measurements + a handful of refinement steps. On the real
broadcast-3n grid the search measured 24 of 27 configs (72 runs vs the grid's
81) and matched the grid's winner at 18/18 sizes. The 3 configs it never
touched (RING/LL/4, RING/LL/40, RING/SIMPLE/4) were never on any contender's
refinement path at any size.

## Why anchor ch=1 is load-bearing

Broadcast/reduce multi-node channel curves are bimodal: ch=1 wins at
32K-512K, high channel counts win elsewhere. With anchors {8,24,48} the climb
starts at ch8 and never looks below, missing the ch=1 winners by up to 53%
(measured 2026-08-18). Anchor set {1,8,24,48} fixed exactly this.
