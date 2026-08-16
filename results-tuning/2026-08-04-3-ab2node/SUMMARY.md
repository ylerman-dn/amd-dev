# 2-node A/B: generated config vs RCCL default

Job 14138, `amd-mi355x-5,7` (both on rail L1 leaves, matching batch 9's pairing), 8 ranks × 1 GPU
per node = 16 ranks. all_reduce, 4K–512M. `validate_tuner_config.py`, 7 repeats per arm
interleaved, keep threshold P(sup) ≥ 0.95 and gain > 2%. Both arms identical
`NCCL_DEBUG=INFO` → file.

Config: `candidate.conf`, md5 `bfaeecc9b5ff038ede405dbe476cf959`, from
`results-tuning/2026-08-04-4-sweep2node/` (165-run sweep, all 6 algo/proto combos, channels
1–48, 0 substituted rows out of 2970).

0/7 hangs both arms.

## Result: 1 of 13 rules kept — a negative result

| size | default | config | delta |
|---|---|---|---|
| 4K | TREE/LL/2 · 0.20 | TREE/LL/2 · 0.20 | +0.0% |
| 8K | TREE/LL/4 · 0.40 | TREE/LL/4 · 0.40 | +0.0% |
| 16K | TREE/LL/8 · 0.79 | TREE/LL/8 · 0.78 | −1.3% |
| 32K | TREE/LL/16 · 1.52 | TREE/LL/16 · 1.51 | −0.7% |
| 64K | TREE/LL/32 · 2.86 | TREE/LL/32 · 2.85 | −0.3% |
| 128K | TREE/LL/48 · 4.95 | TREE/LL128/16 · 4.81 | −2.8% |
| 256K | TREE/LL/48 · 8.56 | TREE/LL128/16 · 8.84 | +3.3% |
| 512K | TREE/LL128/48 · 16.87 | TREE/LL128/24 · 16.30 | −3.4% |
| 1M | TREE/LL128/48 · 30.46 | TREE/LL128/24 · 29.50 | −3.2% |
| 2M | TREE/LL128/48 · 54.88 | TREE/LL128/32 · 52.72 | −3.9% |
| 4M | TREE/LL128/48 · 94.16 | TREE/LL128/40 · 90.99 | −3.4% |
| 8M | TREE/LL128/48 · 144.86 | TREE/LL128/40 · 139.66 | −3.6% |
| 16M | TREE/LL128/48 · 194.99 | TREE/LL128/48 · 194.03 | −0.5% |
| **32M** | TREE/LL128/48 · 222.23 | **RING/LL128/40 · 246.88** | **+11.1%** |
| 64M | RING/LL128/48 · 292.20 | RING/LL128/48 · 291.11 | −0.4% |
| 128M | RING/SIMPLE/48 · 322.73 | RING/SIMPLE/32 · 311.24 | −3.6% |
| 256M | RING/SIMPLE/48 · 374.24 | RING/SIMPLE/32 · 361.80 | −3.3% |
| 512M | RING/SIMPLE/48 · 379.07 | RING/SIMPLE/32 · 375.51 | −0.9% |

Only `32M, ring/ll128, 40ch` survived: **+11.1%**, P(sup) = 1.00. It is an **algorithm** change —
default picks TREE/LL128 at 32M, RING/LL128 is faster there. Not a channel tweak.

## Why 2 nodes has almost no headroom, and 1 node had a lot

Two distinct causes, both visible in the table:

**1. At 4K–64K the config's picks are byte-identical to default** (`TREE/LL` at 2/4/8/16/32
channels). Hence the literal +0.0% rows. RCCL's 2-node defaults already ramp channels the way our
sweep would choose. Contrast 1 node, where default uses 1,1,1,2,4,8,16 channels at those sizes and
pushing them up won +12% to +63%.

**2. From 128K up, default sits at 48 channels — the multi-node cap — and our config picked
fewer** (16/24/32/40). That cost ~3% at eight sizes. The cause is `optimize_metrics.py`'s
tie-break: among configs within 5% of the best it prefers the **fewest channels**. When default is
already at the ceiling, that tie-break can only move downward, and it trades ~3% bandwidth away.

That tie-break is defensible in principle — each channel costs GPU compute units the model could
use, which is the same motive as WarpSpeed — but it is a *bandwidth-for-CUs* trade that
`validate_tuner_config.py` correctly rejects when scored on bandwidth alone. If CU efficiency is
ever the objective, the scoring has to change to reflect it; right now we score only bandwidth.

## No landmine here, and that is explainable

The 1-node config lost 65.7% at ≥64M to a plugin channel-unit mismatch under WarpSpeed's ×4
multiplier. That cannot happen at 2 nodes: **WarpSpeed never engages** — 0 `RING*` rows across all
2970 sweep rows, consistent with batch 9. With no multiplier, plugin `channels=N` and env-var
`NCCL_*_NCHANNELS=N` mean the same thing, so the config deploys what the sweep measured. Worst
case here is −3.9%, not −65.7%.

## Caveats

- Sweep busbw (mpirun, outside Slurm) is not directly comparable to A/B busbw (srun). The A/B is
  self-contained.
- Sweep spread at 2 nodes: median 0.8%, worst 80%, 72 of 990 points above 5%. Multi-node retains
  the occasional startup stall seen in batch 9; the A/B's 7 interleaved repeats and rank statistic
  are what protect the verdicts from it.
- Co-tenants shared the leaf switches during these runs. That can affect absolute busbw but not the
  A/B comparison, since both arms ran interleaved under the same conditions.

## Files

`presentation.html`, `candidate.conf`, `candidate.validated.csv`, `validate_report.txt`,
`times.csv`, `logs/` (238 files).
