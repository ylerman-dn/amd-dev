# GPU-107 findings

Confirmed only — each entry is backed by a captured log or a primary source, and cites the run it
came from. Hypotheses live in the run `SUMMARY.md` files, not here.

8 ranks × 1 GPU per node, RCCL 2.28.3-develop:2e42aa8. Finding 02 covers 1, 2 and 3 nodes;
01 and 03 are single-node unless stated.

| # | finding |
|---|---|
| [01](01-verifying-what-ran.md) | How to verify what RCCL ran — `-A 1`, and why it can't be trusted for channels |
| [02](02-per-collective-defaults-and-accepted.md) | **Defaults and accepted combinations, 6 collectives × 1/2/3 nodes** |
| [03](03-logging-affects-busbw.md) | Logging shifts busbw — never compare logged against unlogged |
| [04](04-pipeline-flow-and-ownership.md) | The sweep→config pipeline, step by step — and the median step that is not in the tool |
| [06](06-optuna-not-viable-here.md) | Optuna (TPE) loses to the racing search, scalarized or fixed to one size |

## Open — deliberately not findings

- **WarpSpeed has never been A/B'd on this hardware.** Its documented benefit is ~50% fewer compute
  units at equal bandwidth — which busbw cannot measure. So we cannot say what it does for us, and
  a busbw-only tuning pass is blind to losing it.
- **`RCCL_DIRECT_ALLGATHER_THRESHOLD` defaults to 72 MiB but the Direct→RING switch is observed at
  ~16M.** Either the comparison uses a different quantity or there are further gates. Untested.
- **Does a tuner config saying `ring` disable WarpSpeed above 64M?** Untested, and busbw would not
  reveal it.
- **Why are channels trimmed to 52/54 at 2M–8M (1 node)?** The binary names a
  `MinTrafficPerchannel` mechanism, but that line never printed in our runs.
- **Why does WarpSpeed never appear multi-node?** No `RING*` at 2 or 3 nodes, any size.
  Not investigated.
- **Does Direct reduce_scatter appear multi-node?** At 1 node the log says "disabled due to PXN
  being disabled" and PXN is multi-node — but we did not check for it in the 2/3-node logs.
- **Why is alltoall unstable multi-node?** 5 of 18 runs died. Not chased.
