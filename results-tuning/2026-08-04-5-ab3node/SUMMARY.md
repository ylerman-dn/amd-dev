# 3-node A/B: generated config vs RCCL default

Job 14138, `amd-mi355x-5,6,7` (5 and 7 on rail L1 leaves, 6 on L2 — matching batch 9's set),
8 ranks × 1 GPU per node = 24 ranks. all_reduce, 4K–512M. `validate_tuner_config.py`, 7 repeats
per arm interleaved, keep threshold P(sup) ≥ 0.95 and gain > 2%. Both arms identical
`NCCL_DEBUG=INFO` → file. Plugin firing verified in all 24 per-rank logs.

Config: `candidate.conf`, md5 `33f167f9470aa0eb02e89e19ec40e023`, from
`results-tuning/2026-08-04-6-sweep3node/` (165-run sweep, all 6 combos, channels 1–48, 0
substituted rows of 2970, 0 `RING*` rows).

0/7 hangs both arms.

## Result: 2 of 13 rules kept

| size | default | config | delta | verdict |
|---|---|---|---|---|
| 4K | TREE/LL/2 · 0.11 | TREE/LL/2 · 0.11 | +0.0% | identical config |
| 8K | TREE/LL/4 · 0.22 | TREE/LL/4 · 0.22 | +0.0% | identical config |
| 16K | TREE/LL/8 · 0.43 | TREE/LL/8 · 0.43 | +0.0% | identical config |
| 32K | TREE/LL/16 · 0.84 | TREE/LL/16 · 0.83 | −1.2% | identical config |
| 64K | TREE/LL/32 · 1.59 | TREE/LL/32 · 1.59 | +0.0% | identical config |
| 128K | TREE/LL/48 · 2.79 | TREE/LL128/8 · 2.78 | −0.4% | drop |
| **256K** | TREE/LL/48 · 4.92 | TREE/LL128/16 · 5.36 | **+8.9%** | **dropped — see below** |
| 512K | TREE/LL128/48 · 10.30 | TREE/LL128/16 · 9.79 | −5.0% | drop (landmine) |
| 1M | TREE/LL128/48 · 18.92 | TREE/LL128/16 · 18.20 | −3.8% | drop |
| 2M | TREE/LL128/48 · 34.96 | TREE/LL128/24 · 33.73 | −3.5% | drop |
| 4M | TREE/LL128/48 · 61.91 | TREE/LL128/32 · 60.16 | −2.8% | drop |
| 8M | TREE/LL128/48 · 103.07 | TREE/LL128/48 · 103.12 | +0.0% | identical config |
| 16M | TREE/LL128/48 · 149.62 | TREE/LL128/48 · 149.12 | −0.3% | identical config |
| **32M** | TREE/LL128/48 · 186.89 | **RING/LL128/32 · 203.09** | **+8.7%** | **KEEP** (algo *and* channels) |
| 64M | RING/LL128/48 · 268.78 | RING/LL128/40 · 268.14 | −0.2% | (in KEEP range) |
| **128M** | RING/LL128/48 · 247.70 | **RING/LL128/40 · 272.92** | **+10.2%** | **KEEP** |
| **256M** | RING/LL128/48 · 297.87 | RING/SIMPLE/32 · 333.85 | **+12.1%** | **dropped — see below** |
| 512M | RING/SIMPLE/48 · 378.26 | RING/SIMPLE/32 · 368.84 | −2.5% | drop (landmine) |

## The most interesting finding: RCCL's 3-node default has a dip at 128M

Default busbw does not rise monotonically with size:

```
 64M  ->  268.78 GB/s
128M  ->  247.70 GB/s   <- 8% LOWER than at half the size
256M  ->  297.87 GB/s
```

Bandwidth should increase with message size. The default runs 48 channels across all three. Our
config's **40 channels at 128M reaches 272.92** — no dip, +10.2%. So the default's 48-channel
choice is specifically bad at 128M, and 40 fixes it.

**Not a noise artefact.** The *slowest* of the seven 64M default repeats (259.99) still beats the
*fastest* of the seven 128M repeats (248.51) — complete non-overlap. It holds in **both**
placements: out-of-place, 64M spans 267.85–269.76 against 128M's 242.18–248.21, again zero overlap.
(The config's *recovery* is strictly monotonic in-place only — out-of-place its 128M r3 reaches
249.81 against a 64M minimum of 254.56, so those overlap. The default's dip is the robust half.)

**Mechanism, from the selection logs.** The 3-node default is identical to the 2-node default at
**16 of 18 sizes**. The only two that differ are **128M and 256M**, where 3 nodes stays on
`RING/LL128` while 2 nodes has already switched to `RING/SIMPLE` — and those are exactly the two
sizes where this config wins (+10.2%, +12.1%). At 256M the win comes from making that protocol
switch earlier than the default does. That is a pattern across the three scales, not a logged
causal statement.

An earlier session saw the same 128M dip, but per this project's rules nothing under `GPU-107/`
counts as verified, so this finding stands on this run's own evidence alone.

## Two genuine wins were rejected by rule granularity, not by measurement

| dropped rule | best gain in range | worst in range | why rejected |
|---|---|---|---|
| `262144-1048576 ch16` | **+8.9%** (at 256K) | −5.0% (at 512K) | landmine |
| `268435456-536870912 ch32` | **+12.1%** (at 256M) | −2.5% (at 512M) | landmine |

Both contain a real improvement *and* a real regression, so `validate_tuner_config.py` correctly
refuses the whole rule — a partial win does not justify shipping a range with a regression in it.

But the rule was only that wide because `generate_tuner_config.py` **merges adjacent sizes that
share identical settings** to keep configs compact. That merge is what buries a +12.1% win next to
a −2.5% loss. Split per size, this run would have kept **4 rules, not 2**: +8.9% at 256K, +8.7% at
32M, +10.2% at 128M, +12.1% at 256M.

**This is a tool limitation, not a hardware result.** Worth fixing: either emit per-size rules and
let the validator prune, or have the validator split a mixed range rather than discard it.

## Scale comparison

| scale | rules kept | gains | why |
|---|---|---|---|
| 1 node | **8 of 12** | +12.5% … +63.0% | defaults use 1–16 channels at small sizes; huge headroom |
| 2 nodes | 1 of 13 | +11.1% | defaults already at the 48-channel cap; only an algo switch helped |
| 3 nodes | 2 of 13 | +8.7%, +10.2% | same, plus a real default anomaly at 128M |

The trend is the result: **RCCL's single-node defaults are loose; its multi-node defaults are
tight.** Where multi-node wins exist they come from algorithm choice (TREE→RING at 32M) or from
correcting a specific default misstep (128M), not from systematically retuning channels.

## Caveats

- Sweep busbw (mpirun, outside Slurm) is not comparable to A/B busbw (srun). The A/B is
  self-contained — both arms srun, interleaved.
- Sweep spread at 3 nodes: median 0.7%, worst 66.7%, 78 of 990 points above 5%.
- **One A/B outlier, kept not dropped:** the default arm's 4M repeat 6 measured **11.69 GB/s**
  against a 61.91 median — 82% spread, the largest in the run (`logs/3n_default_r6.log`). It does
  not change the 4M verdict (P(sup) 0.27, dropped anyway) because P(sup) is a rank statistic, but it
  is why this run must not be read off means. Every other size's spread is under 8%.
- Only **four** sizes are exact ties (4K, 8K, 16K, 64K); 8M is +0.05%, which merely *prints* as
  +0.0%. Seven rules were dropped for "best gain only", six of which asked for what the default was
  already doing.
- Co-tenants shared leaf switches during these runs; affects absolute busbw, not the A/B contrast.
- 7 of 18 sizes had a config byte-identical to default, so they could not differ by construction.

## Files

`presentation.html`, `candidate.conf`, `candidate.validated.csv`, `validate_report.txt`,
`times.csv`, `logs/` (14 stdout + per-rank INFO logs).
