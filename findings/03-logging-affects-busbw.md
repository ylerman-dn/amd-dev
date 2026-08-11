# Logging shifts busbw — never compare logged against unlogged

INFO to **stdout** is slower at every size. INFO to a **file** is free.

| size | INFO→stdout | INFO→file |
|---|---|---|
| 8K | −4.0% | +1.1% |
| 1M | −2.1% | −0.4% |
| 16M | −0.7% | −0.1% |
| 512M | −0.1% | +0.0% |

INFO→file vs no-logging, across all 18 sizes: mean −0.06%, range −0.71% to +1.36%, and the
5-repeat min–max ranges **overlap at 18 of 18 sizes**. Free within our measurement resolution —
not proven to be exactly zero.

The cost is stdout contention, not logging: both variants emitted 51,551 INFO lines; only the
destination differed.

**Rule.** Every run that will be compared must use an identical logging configuration. Use
`NCCL_DEBUG_FILE=<path>_%p.log` (the `%p` stops 8 ranks sharing one descriptor); never INFO on
stdout; never mix logged and unlogged runs in one comparison. The bias is size-dependent so it
does not cancel, and `optimize_metrics.py` discriminates at a 5% tolerance — the same order as
the bias.

3 variants × 5 repeats, interleaved · `results-tuning/2026-08-03-5-lognoise/`
