# 2026-08-19-1 — A/B validation of the multi-node broadcast/reduce grid winners

The 2026-08-18 grids claimed RING/LL128 with **1 channel** beats every higher channel count at
32K–512K for broadcast and reduce at 2 and 3 nodes. This run A/B-validates those winners against
RCCL default through the deployed tuner plugin.

Job 20670 (own alloc {5,6,7}, -J ylerman-bcmn, 10:46–11:43Z, released after). 2n on {5,7},
3n on {5,6,7}. `validate_tuner_config.py` (deployed copy, md5 f626e5b2, identical to
`tools/rccl-sweep/validate_tuner_config.py`): `--repeats 7 --warmup-runs 4|8 (2n|3n)
--plugin-must-fire --split-ranges --min-bytes 4096 --max-bytes 536870912 --iters 20 --warmup 5`,
preflight 3 (default), plugin `librccl-tunerv4-dn.so`, env from the deployed
`rccl-sweep-optuna/sweep_config.yaml` (22 vars), INFO→file both arms. Queue had no multi-node
co-tenant at any point; nodes 5–7 were exclusively ours.

Config generation: per-repeat grid CSVs from `results-tuning/2026-08-18-1-newcolls/` →
`merge_metrics.median_rows` (`*_median.csv` here) → `optimize_metrics.py -t 0.5` (`*_optimized.csv`)
→ `generate_tuner_config.py --include-algo-proto` (`*.conf`). Full-range configs (6–13 rules each);
the A/B judges every rule.

## Kept rules (validated)

| collective | scale | rule | best gain | P(sup) | log |
|---|---|---|---|---|---|
| reduce | 2n | `32768-262144 ring/ll128/1` | **+132.1%** | 1.00 | `reduce_2n/validate.log` |
| reduce | 2n | `16384 ring/ll128/24` | +8.2% | 0.97 | " |
| reduce | 2n | `268435456 ring/simple/40` | +4.2% | 1.00 | " |
| reduce | 2n | `536870912 ring/simple/32` | +2.4% | 1.00 | " |
| reduce | 3n | `32768-524288 ring/ll128/1` | **+105.3%** | 1.00 | `reduce_3n/validate.log` |
| reduce | 3n | `16777216-536870912 ring/ll128/48` | +7.6% | 1.00 | " |
| broadcast | 3n | `32768-524288 ring/ll128/1` | **+137.8%** | 1.00 | `broadcast_3n/validate.log` |
| broadcast | 3n | `67108864-536870912 ring/ll128/48` | +10.2% (worst in-range −1.5%) | 1.00 | " |

Shippable outputs: `reduce_2n.validated.csv`, `reduce_3n.validated.csv`,
`broadcast_3n.validated.csv` (also on `/opt/shared/ylerman/GPU-107/bcmn-2026-08-19/`).

## The headline — ch=1 at 32K–512K is real, and the grids predicted it accurately

Per-size measured gain (median of 7 vs 7) against grid-predicted gain (`predicted_gains.csv`,
grids measured on different nodes {1,3}/{1,3,6}):

```
                     32K      64K      128K     256K     512K
reduce    2n meas  +30.7%  +113.6%  +132.1%   +38.6%   (excl.)
          2n pred  +31.9%  +115.4%  +140.0%   +39.8%
reduce    3n meas  +17.2%   +44.7%  +101.9%  +105.3%   +16.6%
          3n pred   +9.3%   +25.4%   +83.2%   +76.3%   +23.2%
broadcast 3n meas  +13.6%  +137.8%  +105.1%   +63.7%   +19.7%
          3n pred  +18.2%  +168.6%  +143.9%  +105.3%   +38.4%
```

(from `*/per_size.csv`; selection verified per run — config arm shows LL128/1 in range, default
shows LL/1..LL128/48, plugin-must-fire enforced on every config run.)

Mechanism visible in the default column of `per_size.csv`: RCCL walks LL/1 → LL/2 → LL/4-5 →
LL128/48 through this region; the LL128/48 hand-off lands far too early and the ch=1 LL128 pipe
beats it by up to 2.3x until ~512K, where 48 channels finally catch up.

## Dropped rules — why

- Mid-size channel tweaks (ch40/32/24 at 512K–128M): −0.1% to −5.8%, P(sup) ≤ 0.45 — grid
  near-ties measured on {1,3}/{1,3,6} did not transfer to {5,7}/{5,6,7}. Correctly refused.
- Small-size rules (4K–16K): gains ≤ +4% with P(sup) ≤ 0.86, or outright regressions.
- broadcast 3n `1M-16M ch32` (grid predicted +7..15%): measured −0.5% best, −5.8% worst. The
  only prediction that flatly failed to replicate; everything else either validated or was a
  near-tie as predicted.

## broadcast 2n: NOT validated — unmeasurable in 15 attempts

14 preflight failures + 1 completed-but-invalid run (rc=2), 10:47–11:42Z:

- 11 of 15 failed on 4K–16K spread (28–262%, usually 8K). 8K default busbw scatters 0.34–0.54
  GB/s run-to-run; the grid session's own {1,3} broadcast defaults show the same (27% @8K,
  83% @4K, 46% @64K in `2026-08-18-1-newcolls/broadcast_2n_def_raw.txt`), while reduce defaults
  are tight (≤12%) — it is broadcast-2n-specific small-size instability, not the fabric and
  not a co-tenant (queue was empty of multi-node jobs; 3n passed at 12% on the same nodes
  minutes later).
- 3 failed on whole-run dips at one large size (512K/4M/512M, up to 262%) — the exporter
  single-run-dip class (findings/05). These are the ones retrying fixes.
- Attempt 12 passed preflight (19.4%) but the A/B itself was invalidated: 8/36 (size,arm)
  points >25% within-arm spread, worst 430% @8K default (`broadcast_2n_invalid12/validate.log`).
  Its verdicts are void; for the record the in-range readout matched the other scales
  (32K–256K ch1 +76.3%, P(sup)=1.00) but is NOT citable, and its `.validated.csv` is kept only
  as `broadcast_2n_invalid12/broadcast_2n.validated.csv.VOID`.

Getting a verdict for broadcast 2n needs a measurement-parameter decision (user approval
required, none taken): e.g. judging spread only over the sizes the config's rules cover that
matter (≥32K), a higher `--max-arm-spread`, or more preflight runs with a median-based gate.
Hypothesis only, labelled as such: the same LL/1-region instability that makes broadcast 2n
default noisy at 4–16K may be what the ch=1 rule exploits at 32K+.

## Caveats

- Grids that generated the configs ran on {1,3} (2n) and {1,3,6} (3n); the A/B ran on the
  published {5,7}/{5,6,7}. The ch=1 wins transferred across node sets; the mid-size near-tie
  winners did not — consistent with node-set specificity or grid-session noise at 3 repeats.
- Within-arm spread points >25% existed in every valid run (2–4 of 36, all at ≤32K or one
  dipped config run), under the 20% invalidity threshold. Verdicts at the affected small sizes
  lean on P(sup), which is rank-based.
- The kept broadcast 3n `67108864-536870912` rule carries a −1.5% @64M inside the range
  (within the 2% limit); its gain lives at 512M where default picks SIMPLE.

## Files

`*_median.csv`, `*_optimized.csv`, `*.conf` (inputs), `predicted_gains.csv`,
`{reduce_2n,reduce_3n,broadcast_3n}/` (validate.log, per_size.csv, times.csv, stdout logs),
`*.validated.csv` (outputs), `broadcast_2n_preflightfail{1..11,13..15}/`,
`broadcast_2n_invalid12/`, `broadcast_3n_preflightfail1/`, `reduce_3n_preflightfail{1,2}/`
(evidence). Full raw runs incl. per-rank NCCL INFO logs (`*_dbg_*.log`) remain on
`/opt/shared/ylerman/GPU-107/bcmn-2026-08-19/`. RUNLOG.md has one row per attempt/run.
