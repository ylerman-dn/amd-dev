# 2026-08-17-4 — Optuna on a single fixed size (replay, no cluster time)

**Question.** The existing Optuna baseline (`2026-08-16-13-optuna-replay/baseline_*.json`)
scores TPE on a scalarized objective (mean of busbw/max across all 18 sizes) and found it
usually never recovers the exact per-size winner (7/10 seeds at 1n, 6/10 at 3n) — a config
that wins at exactly one size looks average once averaged with 17 others, so Optuna has no
signal to chase it. Does removing the averaging - fixing one size and letting Optuna optimize
that single scalar directly - fix this?

**Method.** `adaptive_search.py baseline --target-size <bytes>` (new flag, this branch):
same TPE/random samplers, same replay mechanics as the existing baseline, but the objective
and the "did it find the winner" check are both restricted to one size instead of all 18.
Backward-compat check: `--target-size` omitted reproduces the committed `baseline_1n.json`
tpe/seed=0 result exactly (configs_to_exact=29, runs_to_exact=87) before any new numbers were
trusted.

Datasets (already measured + verified, committed on `gpu107-optuna`, reused unchanged):
`results-tuning/2026-08-04-1-sweep1node/merged.csv` (1 node, 30 configs),
`results-tuning/2026-08-16-12-variance/variance.json` (2 and 3 node, 54 configs each).

Sizes tested: 4K, 64K, 1M, 16M, 256M (spanning the noisy small-size band and the
bandwidth-bound large-size band noted in `2026-08-17-1-optuna1node/SUMMARY.md`).
Samplers: TPE and random, 10 seeds each, tol 0.5% (project standard). 3 node counts x 5
sizes x 2 samplers x 10 seeds = 300 replay trials, all offline.

## Results

Per (nodes, size): median configs a sampler must evaluate before its picked winner (from
configs seen so far) exactly matches the full-grid winner at that size. Full grid = 30
configs at 1n, 54 at 2n/3n.

**Every one of the 15 (nodes, size) combinations: both TPE and random found the exact
winner in 10/10 seeds.** (vs 7/10 and 6/10 for the OLD scalarized objective.) Full table:
`aggregate.json`, raw per-seed data: `n{1,2,3}_sz{size}.json`.

TPE needs fewer configs (lower median) than random in 12/15 combos, ties in 2, random wins
in 1 (1 node, 1M: random median 10 vs TPE 15). Average median across all 15: TPE 17.6
configs, random 22.4 configs (full grids are 30/54).

## Conclusion

Fixing the size does fix the reliability problem: 100% exact-winner recovery vs ~65% with
the scalarized objective, in this replay data. TPE's edge over blind random is real but
modest (fewer configs in 12/15, not dramatically fewer) — the big win here is removing the
averaging, not the sampler.

**Caveat: replay only**, same as the original optuna-replay phase — bounded by what these
three already-measured grids contain. Not yet validated live.

## Reproduce

```
python3 tools/rccl-sweep/adaptive_search.py baseline \
  --dataset results-tuning/2026-08-04-1-sweep1node/merged.csv \
  --samplers tpe,random --seeds 10 --target-size 4096

python3 tools/rccl-sweep/adaptive_search.py baseline \
  --dataset results-tuning/2026-08-16-12-variance/variance.json --nodes 2 \
  --samplers tpe,random --seeds 10 --target-size 16777216
```

Code change: `tools/rccl-sweep/adaptive_search.py` — `run_sampler_baseline()` gained
`target_size=None`; when set, `obj()` returns raw busbw at that size instead of
`scalarize()`, and the exact/gap checks run against `[target_size]` instead of all 18.
`cmd_baseline` / argparse gained `--target-size`.
