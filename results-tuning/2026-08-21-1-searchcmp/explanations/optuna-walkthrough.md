# How the Optuna (TPE) baseline works — worked example, real trajectories

Same space as the adaptive walkthrough: broadcast 3n, 27 configs. The grid's
winners (per size, median-of-3, 0.5% tie rule) are the ground truth.

## Protocol (findings/06)

TPE proposes one config at a time; we hand it that config's measured score
(a normalized average across the 18 sizes - TPE optimizes one scalar, not 18
objectives). After each NEW config it proposes, we freeze its evaluated set
and check: do the winners selected from that set match the grid's winners at
ALL 18 sizes? The first count where the answer is yes is that seed's cost.
Duplicate proposals are answered from cache and cost nothing.

## Real trajectory, seed 2 (matched)

First 8 proposals: SIMPLE/40, LL128/2, LL128/24, LL/2, LL128/40, LL/24,
SIMPLE/8, LL128/48. After the 16th unique config, its evaluated set contained
every per-size winner, and the check passed: cost = 16 configs (48 runs).
To get those 16 unique configs it made 810 proposals - 788 were repeats of
configs it had already tried.

## Real trajectory, seed 1 (never matched)

810 proposals, 784 of them repeats. It evaluated 26 of the 27 configs; the
single config it never proposed was RING/LL/24 - which happens to be the
grid winner at size 4K. Without it, the winner set can never match, at any
budget. Cost = infinite.

## The two honesty caveats

1. The stop-point is oracle knowledge. WE check its winners against the grid
   truth after every step; Optuna itself has no signal that says "you now have
   all 18 winners". A real deployment must pre-commit a budget and hope.
2. The duplicate storm (788/810) is not a harness bug: TPE has no built-in
   de-duplication on small categorical spaces and keeps re-proposing the
   configs it already believes are good. We cap proposals at 27x30 = 810 to
   bound the loop; duplicates are free in our cost model, so the numbers shown
   are TPE's best case. It still loses to the racing search.

## Random control

Same protocol, but the "proposal order" is just a shuffle of all 27 configs.
It always matches eventually (a shuffle is exhaustive); its median cost across
10 seeds (~23.5 configs of 27) sits barely below the full grid - proving that
on this problem, clever proposal ordering is nearly worthless: per-size winners
are scattered (LL/24 at 4K, LL128/1 at 32K-512K, LL128/48 at large sizes), so
you must evaluate most distinct region-winners before all 18 line up. The
racing search wins not by ordering but by CHEAP PARTIAL INFORMATION: anchor
measurements sketch whole channel curves at all sizes at once.
