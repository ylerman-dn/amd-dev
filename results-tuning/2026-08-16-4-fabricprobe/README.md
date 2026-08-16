# Fabric contention probe — 2026-08-16

Question: can a multi-node A/B produce a decidable answer right now?

Method: run the DEFAULT configuration 5 times at 2 nodes (`amd-mi355x-5,7`, job 14457) and look at
how much its own repeats disagree. Nothing is compared — identical runs should give identical
numbers, so any spread is the environment.

```
size   5 repeats of the SAME config              median   spread
  1M   11.78  14.67  10.87  15.92  12.55          12.55    46.5%
  8M   49.96  27.48  70.68  85.29  66.71          66.71   210.4%
 64M  172.62 187.51 109.80 111.13 188.81         172.62    72.0%
```

**Answer: no.** Cause: co-tenant job 14456 (`hai_test2`, user `amd`, nodes 4/6/8/9) saturating the
shared spine. Our nodes were clean — no stray processes, no competing steps.

Cost of knowing: 40 seconds, versus ~5 minutes per scale for an A/B that has to be discarded.

This probe is now built into the tool as `validate_tuner_config.py --preflight N` (default 3), which
aborts with exit 3 before spending the A/B. Verified against this same fabric: it reported
`2n: WORST SPREAD 378% at 2097152 bytes (limit 25%) -- NOT usable` and refused to continue.

Raw: `r1.log`–`r5.log` (probe), `preflight_times.csv` (the built-in version).
