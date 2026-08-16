# Exact command chain — 2026-08-04 sweep → config → A/B (all scales)

Covers 1, 2 and 3 nodes. Stages 0–6 below are written from the 1-node run; the
"All three scales" section near the bottom gives the per-scale differences.

Node `amd-mi355x-7`, Slurm job **14138** (holds nodes 5,6,7, `-t 480`, expires 20:05Z).
Everything runs on the cluster; `/opt/shared` is not mounted on the dev VM.
`$SW = /opt/shared/ylerman/GPU-107/rccl-sweep` (the fixed tool, copied from `tools/rccl-sweep/`)

## Stage 0 — tool fixes (dev VM, then copied to cluster)

| file | change |
|---|---|
| `sweep_executor.py` | `-M 1` → `-A 1`; `-R` → `-C`; added optional `-Z csv -X` |
| `rccl_sweep.py` | record **measured** algo/proto, flag `substituted` when request not honoured |
| `sweep_db.py` | +6 audit columns, additive migration, carry them into the CSV export |
| `optimize_metrics.py` | drop `substituted == 1` rows; **rank on `busbw_ip`, not `time_ip_us`** |
| `unsupported_combos.yaml` | encode the measured honoured-combination map |
| `sweep_config.yaml` | `iterations: 5` → **20**; `report_cputime: 1` → **0** |
| `validate_tuner_config.py` | `-M 1`→`-A 1`, drop `-R 1`, 8 ranks × 1 GPU (was 1 × 8), `--jobid`/`--nodelist`, `--plugin-must-fire` |

Deploy to cluster:
```
cd tools && tar czf /tmp/rccl-sweep.tgz --exclude=__pycache__ --exclude=sweep_results rccl-sweep
scp /tmp/rccl-sweep.tgz amd-mi355x-1:/tmp/
ssh amd-mi355x-1 'cd /opt/shared/ylerman/GPU-107 && rm -rf rccl-sweep && tar xzf /tmp/rccl-sweep.tgz'
```

## Stage 1 — allocation

```
salloc --no-shell -p XAI -N3 -w amd-mi355x-5,amd-mi355x-6,amd-mi355x-7 \
       --gres=gpu:8 -t 480 -J ylerman-ab     # -> job 14138
```
1-node = node7 · 2-node = {5,7} (both L1 leaves) · 3-node = {5,6,7} (2×L1 + 1×L2).
Topology deliberately matches batch 9 so the scales stay comparable.

## Stage 2 — smoke gate (proves the fixes work)

```
cd $SW && export MY_PATH=/opt/shared/ylerman/GPU-107/bin
echo "172.30.160.165 # amd-mi355x-7" > servers.txt
python3 rccl_sweep.py --servers servers.txt --output-dir <out> \
  --nodes 1 --collective all_reduce --channels 32 \
  --algo RING,TREE --proto SIMPLE --min-size 256M --max-size 512M
```
Expected: `RING/SIMPLE` honoured; `TREE/SIMPLE` → `substituted: 2/2 ... (measured RING*/SIMPLE)`.

## The benchmark args are NOT all on the command line

`rccl_sweep.py` takes sizes/channels/algo/proto as flags, but `-n` (iterations), `-w` (warmup),
`-c` (check) and `-A`/`-C` come from **`sweep_config.yaml`**. So the invocations below do not show
`-n` at all. The effective command the tool actually built, taken from a run's own `command.txt`:

```
# superseded -n 5 sweep
all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 5  -w 5 -c 1 -A 1 -C 1
# superseded -n 20 sweep (still had -C 1)
all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1 -C 1
# FINAL: -C dropped (report_cputime: 0)
all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1
```

`-A 1` replaces the wrong `-M 1`. **`-C 1` was then also removed**: `-C` is
`--report_cputime`, which makes the time column report CPU launch time (~constant regardless of
collective duration) — so `optimize_metrics.py`, which originally ranked on `time_ip_us`, was
ranking on near-noise and selected 1-channel configs that were 99% slower than default. The fix
was both: `report_cputime: 0` in `sweep_config.yaml`, and rank on `busbw_ip` instead.

Every run's exact command is preserved at `<run>/outputs/<case>/command.txt` — check there rather
than trusting this file.

## Stage 3 — the sweep

`driver.sh` (kept in the run folder) runs, for each of 3 repeats:
```
python3 rccl_sweep.py --servers servers.txt --output-dir $OUT/rep$r \
  --nodes 1 --collective all_reduce --min-size 4K --max-size 512M \
  --channels 1,2,4,8,16,24,32,40,48,56 --algo RING --proto SIMPLE,LL   # 20 runs
  ... same with --algo TREE --proto LL                                  # 10 runs
  ... same with no --channels/--algo/--proto                            #  1 run (default)
```
= 31 runs/repeat × 3 = **93 runs**. Launched detached (`setsid nohup ./driver.sh &`);
progress appended to `$OUT/PROGRESS`.

## Stage 4 — merge → median → optimize → config

```
# flatten so merge_metrics.py (one level deep) sees all 9 sessions
for d in $OUT/rep*/run_*; do ln -s "$d" $OUT/flat/run_NN_repN; done
python3 merge_metrics.py --base-path $OUT/flat -o $OUT/merged.csv
# NOTE: median-per-(config,size) aggregation is inline python, NOT part of the tool.
#       Without it optimize_metrics.py picks the single luckiest run out of 90 rows/size.
python3 <inline>  merged.csv -> merged_median.csv   (median time_ip_us + spread_pct)
python3 optimize_metrics.py $OUT/merged_median.csv -o $OUT/optimized.csv
python3 generate_tuner_config.py $OUT/optimized.csv -o $OUT/generated.conf --include-algo-proto
```

## Stage 5 — A/B (done; see the all-scales section below for the real invocation)

```
python3 validate_tuner_config.py --config <generated.conf> \
  --binary /opt/shared/ylerman/GPU-107/bin/all_reduce_perf \
  --plugin /opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so \
  --repeats 7 ...
```
Gate before trusting any A/B number: `TUNER/Plugin: Using ... (v4)` must appear in the INFO log,
**and** the `-A 1` selection must change in the direction the config asked for.

## Superseded

`sweep-2026-08-04-n5/` — first attempt at `-n 5`. Kept as evidence: 10–20% median repeat spread,
which is larger than `optimize_metrics.py`'s 5% equivalence tolerance, so its min-channels
tie-break selected 1-channel configs at 16M–64M. Its `generated_n5.conf` is visibly broken.

---

# All three scales — 2026-08-04

Each scale is two folders: sweep (config creation) then A/B (config testing).

| scale | sweep folder | A/B folder | nodes |
|---|---|---|---|
| 1 node | `2026-08-04-1-sweep1node` | `2026-08-04-2-ab1node` | 7 |
| 2 nodes | `2026-08-04-4-sweep2node` | `2026-08-04-3-ab2node` | 5,7 (both L1 leaves) |
| 3 nodes | `2026-08-04-6-sweep3node` | `2026-08-04-5-ab3node` | 5,6,7 (2×L1 + 1×L2) |
| final | — | `2026-08-04-7-validated` | combined config |

(The folder numbers interleave because each A/B was run as soon as its config was ready, before the
next scale's sweep finished.)

## Sweep — differences between scales

```
# 1 node: only the 3 algo/proto combos honoured at 1 node, channels up to 56
--nodes 1 --channels 1,2,4,8,16,24,32,40,48,56 --algo RING --proto SIMPLE,LL   # 20 runs
--nodes 1 --channels <same>                    --algo TREE --proto LL          # 10 runs

# 2 and 3 nodes: ALL 6 combos are honoured, and channels cap at 48
--nodes N --channels 1,2,4,8,16,24,32,40,48 --algo RING --proto SIMPLE,LL,LL128  # 27 runs
--nodes N --channels <same>                 --algo TREE --proto SIMPLE,LL,LL128  # 27 runs
```
Plus one unforced default run per repeat. 3 repeats each.
93 runs at 1 node, 165 at 2 and 3 nodes. The matrix differs because the *acceptance map* differs
(`findings/02`), not because of node count.

Multi-node also needs the fabric env, which comes from `sweep_config.yaml` (`NCCL_IB_HCA`,
`NCCL_SOCKET_IFNAME`, `NCCL_IB_GID_INDEX`, …). Without it 2+ node runs produce no output at all.

## A/B — same for every scale

```
python3 validate_tuner_config.py \
  --config <candidate.conf> \
  --binary /opt/shared/ylerman/GPU-107/bin/all_reduce_perf \
  --plugin /opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so \
  --jobid 14138 --nodelist <nodes> --ranks-per-node 8 --gpus-per-rank 1 \
  --repeats 7 --min-bytes 4096 --max-bytes 536870912 --iters 20 --warmup 5 \
  --plugin-must-fire --logdir <logs> --times-csv <times.csv> \
  --env NCCL_SOCKET_IFNAME=... --env NCCL_IB_HCA=... (multi-node only)
```

`--plugin-must-fire` is a gate I added: a run only counts if its INFO log contains
`TUNER/Plugin: Applied config`. Without it, a config that silently failed to load would be A/B'd
against itself and reported as "no difference".

`validate_tuner_config.py` needed four fixes before it could be used at all
(`tools/rccl-sweep/validate_tuner_config.py`): `-M 1` → `-A 1`, drop `-R 1`,
`--ntasks-per-node=1 -g 8` → `8 ranks × 1 GPU` (it would otherwise have measured a different
topology from the sweep that produced the config), and `--jobid`/`--nodelist` so it runs inside the
existing allocation.

## Two traps worth remembering

**Freezing the tool per run freezes its bugs too.** Each run folder holds `tool/`, a frozen copy —
which protects a running sweep from a mid-flight redeploy (that killed one sweep with a stale file
handle). But the 2-node A/B then failed immediately because its frozen copy predated the
`validate_tuner_config.py` patch. Copy the patched file in, or re-freeze, before each A/B.

**`optimize_metrics.py` ranks on `busbw_ip`, not `time_ip_us`.** With `-C/--report_cputime` the time
column reports CPU launch time (~constant regardless of collective duration), so ranking on it
selects near-arbitrary configs — it produced 1-channel rules that were 99% slower than default.
`report_cputime` is now `0` in `sweep_config.yaml` and the ranking is on bandwidth.
