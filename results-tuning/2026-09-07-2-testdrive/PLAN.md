# PLAN — test drive of the two processes, scripts only (2026-09-07)

Goal: run Process A (sweep -> conf -> A/B) and Process B (conf in Qwen3-30B-A3B) end-to-end
with the pruned tool at repo HEAD eeb5aaa, one node, the operator only allocating and
launching one command per step.

## Node, allocation, tool copy
- Node: amd-mi355x-9 (idle; the only idle node holding Qwen3-30B-A3B locally at
  /huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/).
  Queue now: only DN_AJ on nodes 6-7. Disk on node 9: 210G free.
- salloc --no-shell -N1 -w amd-mi355x-9 --gres=gpu:8 -t 480 -J ylerman-testdrive   (from amd-mi355x-1)
- Tool copy: scp repo tools/rccl-sweep (HEAD eeb5aaa) -> node9:/data/ylerman/testdrive-2026-09-07/tool/
  servers.txt = "amd-mi355x-9". Hash recorded in this dir's tool_commit.txt.
- Every step launched as:  srun --jobid=$JOBID -N1 bash -c '<command>' </dev/null > <step>.driver.log 2>&1 &

## Process A

### A2 sweep (1 pass)
cd /data/ylerman/testdrive-2026-09-07/tool && python3 rccl_sweep.py --servers servers.txt \
  --collective all_reduce --nodes 1 --algo RING,TREE --proto LL,LL128,SIMPLE \
  --channels 1,2,4,8,12,16,24,32,40,48,56,64,72,80,84,96,104,112 \
  --min-size 4K --max-size 512M --output-dir /data/ylerman/testdrive-2026-09-07/out

Grid: all_reduce x 1 node x {RING,TREE} x {LL,LL128,SIMPLE} x 18 channel values = 108 cells,
18 sizes 4K..512M (x2 steps). 1 repeat (test drive; merge --median is then a no-op).
Env per cell (sweep_config.yaml): runtime container, image lmsysorg/sglang:v0.5.17-rocm720-mi35x,
RCCL_MSCCL_ENABLE=0, NCCL_MIN/MAX_NCHANNELS=<cell>, NCCL_ALGO/NCCL_PROTO=<cell>,
NCCL_DEBUG=INFO to dbg_%p.log, -n 20 -w 5 -c 1 -A 1. Image NCCL_MIN_NCHANNELS=112 is
OVERRIDDEN per cell by the channel value (not dropped).
Expected: TREE/SIMPLE cells abort rc=3 (invalid on this arch, seen 2026-09-07-1); LL128 cells
are expected to be substituted by RCCL and dropped by optimize_metrics (this is the check).
ETA: 44 cells took ~25 min on 2026-09-07-1 -> ~60 min.

### A3 pick winners
cd /data/ylerman/testdrive-2026-09-07/tool && \
python3 merge_metrics.py --base-path /data/ylerman/testdrive-2026-09-07/out --median --exec-from-logs \
  -o /data/ylerman/testdrive-2026-09-07/merged.csv && \
python3 optimize_metrics.py /data/ylerman/testdrive-2026-09-07/merged.csv \
  -o /data/ylerman/testdrive-2026-09-07/optimized.csv -t 0.5
Tolerance 0.5% = script default and the value used for gpu107_1n_v4_scripted (RUNLOG 2026-09-07T13:5xZ).

### A4 initial conf
python3 generate_tuner_config.py /data/ylerman/testdrive-2026-09-07/optimized.csv \
  -o /data/ylerman/testdrive-2026-09-07/testdrive_1n.conf --include-algo-proto

### A5 A/B vs RCCL default
cd /data/ylerman/testdrive-2026-09-07/tool && python3 validate_tuner_config.py \
  --config /data/ylerman/testdrive-2026-09-07/testdrive_1n.conf \
  --binary /opt/shared/ylerman/GPU-107/bin/all_reduce_perf \
  --plugin /opt/shared/ylerman/GPU-107/infer-2026-08-30/librccl-tunerv4-dn.so --plugin-must-fire \
  --runtime container --drop-env NCCL_MIN_NCHANNELS \
  --repeats 7 --min-bytes 4096 --max-bytes 536870912 --iters 20 --warmup 5 --split-ranges \
  --logdir /data/ylerman/testdrive-2026-09-07/ab
Arms: default = image env with NCCL_MIN_NCHANNELS REMOVED, MSCCL=0, no plugin.
      config  = same env + plugin + testdrive_1n.conf. 7 reps/arm interleaved.
Gates (script defaults): P(sup) >= 0.95, gain > 2%, regression < 2%, preflight 3 runs,
warmup-runs 4, noise-gate spread 25%. Same setup as RUNLOG 2026-09-07T15:0xZ (ab_v4_noflag).
Output: testdrive_1n.validated.csv (or .VOID-rc2). ETA ~50 min for ~9 rules.

### A6
Copy /data/ylerman/testdrive-2026-09-07/{merged,optimized}.csv, testdrive_1n.conf,
testdrive_1n.validated.csv, ab/per_size_stats.csv, ab/resolved_env.txt, out/run_*/metrics.csv,
*.driver.log into this dir.
HTML page: NOT scripted -> written by me after the run, every number cites a file here.

## Process B (Qwen3-30B-A3B, node 9, same allocation)
cp testdrive_1n.validated.csv /opt/shared/ylerman/GPU-107/infer-2026-08-30/testdrive_1n.validated.conf
IM_DROP_FLOOR=1 IM_CONF_DIR=/opt/shared/ylerman/GPU-107/infer-2026-08-30 \
IM_BASE=/data/ylerman/testdrive-2026-09-07/model \
  bash /data/ylerman/testdrive-2026-09-07/tool/infer_paired.sh \
  /huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/ \
  td_q 10 rule:testdrive_1n.validated.conf none:NONE
One server, TP-8, plugin hot-reload build, RCCL_MSCCL_ENABLE=0, --disable-custom-all-reduce,
NCCL_MIN_NCHANNELS REMOVED (matches A5's baseline), RM_SUBSYS=INIT,ENV (quiet), ISL/OSL 512/512,
conc 32, 128 prompts. 10 rounds x {rule, none} interleaved. "none" = plugin loaded, zero rules.
ETA ~45 min incl. server start.
B3 scorer: NOT scripted. Proposed: new tools/rccl-sweep/infer_score.py (tok/s per round per arm,
drop round 1, P(sup) + median gain, same gates as A5). Written and shown before B runs, or B's
verdict is computed by hand and labelled so.

## Receipts
RUNLOG row at each launch; command.txt per cell (tool); this PLAN.md; tool_commit.txt;
the runs' own dbg_*.log (A) and rccl.*.log (B).
