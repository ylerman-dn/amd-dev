# Why Optuna loses here (and where it would win)

## Verdict on this problem

- Space is tiny (20-54 configs per grid) and trials are cheap (~10s). You can
  afford to measure your way to certainty; guessing adds nothing. TPE's median
  cost to the exact winners barely beat a random shuffle (e.g. broadcast 3n:
  ~20.5 vs ~23.5 of 27 configs).
- TPE has no de-duplication on small categorical spaces: real trace, seed 2 -
  810 proposals, 788 repeats. Seed 1 never proposed RING/LL/24 at all (the 4K
  winner), so it could never finish.
- It optimizes ONE scalar, so 18 per-size objectives must be averaged
  (scalarized), which blurs exactly the per-size distinctions we tune for.
- Inherent flaw for our use: no stopping signal. "Matched after N configs" is
  known only by comparing against the grid truth - which a real deployment
  does not have. A budget must be fixed blind.
- Under a relaxed bar (within 5% of the winner) optuna converges in ~all seeds
  at ~half the cost - but the stopping problem remains, and by the same
  relaxation the racing search gets cheaper too (1+2 policy: 25-45% fewer runs
  at worst 2.7% gap). One bar must be applied to all methods.

## Where Optuna is the right tool

Huge spaces (thousands+ combos), expensive trials, no ground truth, success =
"best found within a fixed budget". If our space ever grows that large
(algo x proto x channels x chunk/threshold knobs), revisit it.

See: findings/06 (original verdict), optuna-walkthrough.md (worked example),
results-tuning/2026-08-21-1-searchcmp/ (all 15 grids, both success bars).
