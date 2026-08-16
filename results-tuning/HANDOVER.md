# Handover — GPU-107, end of 2026-08-16

State of the work, so anyone (including a future session) can pick up without re-deriving.

---

## What is verified and shippable

A tuner config that beats RCCL's default, A/B-tested through the plugin, 7 repeats per arm,
probability-of-superiority 1.00 on every rule.

| scale | rules | gains | config |
|---|---|---|---|
| 1 node | 7 of 12 | +9.4% … **+65.3%** across 4K–1M | `2026-08-16-3-ab1node-valid/gen_1n_t1.conf` |
| 2 nodes | 3 of 13 | +6.6% @256K, +9.5% @32M, +3.3% @128M | `2026-08-16-10-ab-t05/gen_2n_t05.conf` |
| 3 nodes | 4 of 12 | +11.7% @256K, +7.2% @32M, +11.8% @64–128M, **+17.3% @256M** | `2026-08-16-10-ab-t05/gen_3n_t05.conf` |

Validated outputs are the `*.validated.csv` next to each — those contain only the rules that passed.

Deploy with:
```
export NCCL_TUNER_PLUGIN=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
export NCCL_TUNER_CONFIG_FILE=<path to a .validated.csv>
```

**Scope:** all_reduce only, MI355X (gfx950), RCCL 2.28.3-develop:2e42aa8, 8 ranks × 1 GPU per node.
Nodes 5,6,7. A different RCCL version invalidates the acceptance map and the defaults being beaten.

---

## Branch and history

- Branch `gpu107-opt2-sweep`, pushed. History squashed from 23 commits to 6 on 2026-08-16.
- `backup-pre-squash-20260816` on the remote holds the pre-squash history. Delete when comfortable:
  `git push origin --delete backup-pre-squash-20260816`
- `results-tuning/` is tracked for evidence (`.md`, `.conf`, `.csv`) but not raw logs (~870 MB stay
  local). `OPEN-ITEMS.md` is deliberately gitignored — it churns.

---

## Two sessions running in parallel

Both were started 2026-08-16 evening, each in its own git worktree, each told to present a plan and
wait for approval before running autonomously.

| worktree | branch | task |
|---|---|---|
| `../amd-dev-optuna` | `gpu107-optuna` | can guided search cut the number of benchmark runs? |
| `../amd-dev-v5` | `gpu107-tuner-v5` | port the plugin v4 → v5, prove parity, then explore constants |

**They share one cluster.** Both were told to check `squeue -p XAI` for a job named `ylerman-ab`
before booking. That is a convention, not a lock. **First thing to check tomorrow: did they collide?**
If both booked nodes at once, their numbers are contaminated — though `--preflight` should have
refused rather than reported noise.

---

## Where everything lives

| what | where |
|---|---|
| **the insights, in plain language** | `results-tuning/INSIGHTS.md` ← start here |
| confirmed facts about RCCL | `findings/` (5 entries, `README.md` indexes them) |
| open items | `results-tuning/OPEN-ITEMS.md` (7 of 9 closed) |
| every command per stage | `results-tuning/COMMANDS.md` |
| every run with env and result | `results-tuning/RUNLOG.md` |
| phase timings | `results-tuning/TIMINGS.md` |
| working rules for this project | `CLAUDE.md` (binding) |
| presentation pages | served at `http://10.10.73.168:8502/` |

Pages are served by a detached `python3 -m http.server` from
`results-tuning/2026-08-10-2-pages/`. It survives a session ending but **not a reboot** — restart with
`cd <that dir> && setsid nohup python3 -m http.server 8502 --bind 0.0.0.0 &`.

---

## What the tool does now that it did not this morning

Nine gates and fixes, all committed:

1. `-A 1` reports the real selection (was `-M`, a memory report)
2. `-R` dropped — it was buffer registration, silently altering performance
3. rows record what RCCL **did**, not what we asked, with a `substituted` flag
4. ranking on bandwidth, not a time column that `-C` could corrupt
5. selection tolerance 5% → **0.5%** (5% was ~7× the measurement noise)
6. `merge_metrics --median` — the repeat-collapse step that previously existed only as shell history
7. `--warmup-runs` — GPUs idle at 195 MHz; cold runs once rejected all 11 rules
8. `--preflight` (exit 3) and a post-run validity gate (exit 2) — a run that cannot decide is
   reported as such instead of as "RCCL's defaults are already optimal"
9. sweep keeps its `NCCL_DEBUG=INFO` logs, so a run can be verified after the fact

Exit codes now mean four things: `0` rules kept, `1` none kept but run valid, `2` run invalid,
`3` preflight failed.

---

## Still open

**#5 is now done** (build-guarded combo rules). Remaining:

- **#9** — the hotspot/adaptive loop finds nothing on these curves. Informational; see INSIGHTS §10.
- **v5 plugin** — the biggest lever, being explored in the `gpu107-tuner-v5` worktree. See INSIGHTS §12.
- **Other collectives** — everything we know is all_reduce. Five untouched.
- **Workload sizes** — we can prove each rule helps, but not what the config is worth, because that
  depends on which message sizes your models actually use.

---

## Traps that cost real time today, so nobody repeats them

- **Two validator processes at once** blocked each other's Slurm step creation and looked exactly
  like a hang. Kill by PID; `pkill -f` patterns miss the full path.
- **`pgrep -f <pattern>` matches your own command line.** It caused three wrong diagnoses in one day.
- **Hand-typing the fabric env** produced a wrong GID index and a truncated HCA list. The A/B now
  reads `env_vars:` from `sweep_config.yaml` instead.
- **`servers.txt` node order matters.** Taking "the first two hosts" gave nodes 5,6 (crossing a spine)
  where our results use 5,7 (same leaf). Not comparable.
- **One waiter per condition.** At one point eight background shells were polling the same file.

---

*Written 2026-08-16. Cluster left free, no allocations held, working tree clean, everything pushed.*
