# A NIC-metrics poller disturbs multi-node runs every ~30 s

`amd-nic-metrics` spawns 9 concurrent `nicctl show rdma queue-pair statistics` processes on each
node, ~40% CPU each, every **~30 s** for **~9 s**. Our benchmark runs take 11–15 s, so roughly a
third of them overlap a poll.

```
12:19:01  2
12:19:03  9      <- 9 nicctl processes
12:19:09  8
12:19:11  0      <- quiet
...
12:19:32 11      <- next cycle, 31 s later
```

Sampled on `amd-mi355x-5`, 2026-08-16, 2 s interval, `pgrep -c nicctl`.

## Why it matters

It is polling **RDMA queue-pair statistics** — the same hardware path the collectives use — not just
burning CPU. Whole runs are affected, not particular message sizes:

```
preflight, 2 nodes, 5 repeats of the SAME default configuration
        1M     2M     4M     8M    16M    32M     64M    128M    256M
 r1   29.4   11.4   14.4    7.2   18.1   11.0    90.2   170.1    38.3   <- hit
 r2   22.8   53.7   52.4   94.0  143.2  163.6   183.2   235.8   325.1   <- partly
 r3    5.1   54.0   89.2   62.3    7.1   20.8   284.4   308.0   227.0   <- hit
 r4   29.8   46.4   88.7  139.9  195.2  221.2   281.8   313.8   369.1   <- clean
 r5   29.7   53.1   89.3  144.3  194.8  219.8   276.1   314.9   368.4   <- clean
```

r4 and r5 agree to ~1%; r1 and r3 are down by up to 10x. Load average on those nodes was 59 and 43
with nothing of ours running.

## What it is not

Not a co-tenant. The same spread appeared with a 4-node job present and after it had finished, and
the topology rules it out anyway: `amd-mi355x-5` and `-7` share an L1 leaf, so a 2-node flow between
them never reaches the spine the co-tenant's L2 nodes use.

Not the GPU clock ramp either, though that is real and separate: warming the GPUs cut 2-node spread
at 64M from 64.8% to 8.0%, and the residual disturbance is what this finding describes.

## Scope

- Measured on `amd-mi355x-5` and `-7`, 2026-08-16. Present on both.
- Applies to **every multi-node measurement in this project**, including the 2026-08-04 sweeps that
  produced the shipped configs — those used 3 repeats, so ~1 in 3 could be disturbed with no way to
  tell after the fact.
- 1-node runs are unaffected in practice: intra-node all_reduce does not use the NICs, and 1-node
  A/B spread is ~1%.

## What would falsify it

Runs taken with `amd-nic-metrics` stopped showing the same spread. Not tested — stopping it affects
everyone on the cluster and was not attempted.

Evidence: `results-tuning/2026-08-16-8-nicmetrics/` · `pgrep`/`ps` output above, and
`/opt/shared/ylerman/GPU-107/ab-2026-08-16/pf3/logs/`
