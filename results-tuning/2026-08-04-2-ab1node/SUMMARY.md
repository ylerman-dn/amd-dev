# 1-node A/B: generated config vs RCCL default

Job 14138, `amd-mi355x-7`, 8 ranks × 1 GPU, all_reduce, 4K–512M.
`validate_tuner_config.py`, **7 repeats per arm, interleaved**, keep threshold P(sup) ≥ 0.95 and
median gain > 2%. Both arms under identical `NCCL_DEBUG=INFO` → `NCCL_DEBUG_FILE` (proven free:
mean −0.06%, `results-tuning/2026-08-03-5-lognoise/`).

Config under test: `candidate.conf`, md5 `dc515bbabd3f57a7120bf11ef01efe1f`, generated from
`results-tuning/2026-08-04-1-sweep1node/` (93-run sweep, `-n 20`, 0 substituted rows).
Deployed via `NCCL_TUNER_PLUGIN=ab-tuner-test/librccl-tunerv4-dn.so`.

0/7 hangs both arms. `#wrong` = 0 on all 252 data rows.

## Result: 8 of 12 rules verified as wins

| size | algo/proto/ch | gain | P(sup) |
|---|---|---|---|
| 4K | tree/ll/2 | +12.5% | 1.00 |
| 8K | tree/ll/4 | +29.1% | 1.00 |
| 16K | tree/ll/8 | +45.3% | 1.00 |
| 32K | tree/ll/16 | +56.1% | 1.00 |
| 64K | tree/ll/32 | **+60.2%** | 1.00 |
| 128K | ring/ll/8 | +32.3% | 1.00 |
| 256K | ring/ll/16 | +39.5% | 1.00 |
| 512K–1M | ring/ll/32 | **+63.0%** | 1.00 |

P(sup) = 1.00 means the config won **every** head-to-head pair of the 49 comparisons at that size.

The mechanism is simple: RCCL's 1-node defaults use very few channels at small sizes
(1,1,1,2,4,8,16 for 4K–256K). Pushing the channel count up is worth 12–63%. That was the one
untested direction — every earlier experiment forced channels *down*, which always hurt.

## 4 rules correctly rejected

| rule | worst | verdict |
|---|---|---|
| 2M–4M ring/simple/32 | −2.7% | landmine |
| 8M ring/simple/48 | −1.2% | insufficient gain |
| 16M–32M ring/simple/56 | −0.4% | insufficient gain |
| **64M–512M ring/simple/48** | **−65.7%** | landmine |

## The −65.7% is a plugin channel-unit mismatch, not a lost algorithm

My first reading was "the config lost WarpSpeed". **That was wrong** — `RING*` and
`WarpSpeed enabled` appear in *both* arms, all 8 per-rank logs. What differs is channel width:

| | requested | base channels | actual under WarpSpeed (×4) |
|---|---|---|---|
| env var `NCCL_MIN/MAX_NCHANNELS=48` (what the sweep measured) | 48 | 48 | n/a |
| plugin `channels=48` (what was deployed) | 48 | **12** | 48 |
| default, unforced | — | 56 | **224** |

Evidence: `-A 1` reports `RING*/SIMPLE/56` for default and `RING*/SIMPLE/12` for the config at
64M–512M, while the plugin's own log says it applied `channels=48`. 48 ÷ 4 = 12.

So the plugin's `channels` field is consumed as the **post-multiplier total**, whereas the env var
sets the **base**. Same integer, 4× different meaning. We deployed ~4.7× fewer channels than
default without intending to.

This is exactly the "measurement path != deployment path" failure `validate_tuner_config.py`'s
docstring warns about, now measured here. It also explains the shape of the whole result: below
64M no multiplier applies, the two paths agree, and that is precisely where all 8 wins are. Every
rule at ≥2M either drew or lost.

**Rule for future configs: channel values taken from an env-var sweep must not be written into a
plugin config unchanged wherever WarpSpeed is active (≥64M for all_reduce, 1 node).**

## The config as written is a net loss

Whole-sweep average busbw: 143.89 → 90.37 GB/s = **−37.2%**. That single landmine rule dominates
the average. The shippable artefact is `candidate.validated.csv` — the 8 surviving rules — not
`candidate.conf`.

## Caveats

- Sweep busbw (mpirun launcher, outside Slurm) runs 1–10% below A/B busbw (srun). The two are
  **not** directly comparable. The A/B is self-contained: both arms use srun.
- The 16K config point has 12% spread across its 7 repeats against a 0.97% median — kept and
  labelled rather than dropped.
- busbw is printed to 2 dp, so at 4K one least-significant digit is 3.1%. The +12.5% gain is wide
  enough to survive that, but small-size percentages carry that floor.

## Files

`presentation.html` (charts + verdicts), `candidate.conf`, `candidate.validated.csv`,
`validate_report.txt`, `times.csv`, `logs/` (126 files: 14 stdout + 112 per-rank INFO logs).
