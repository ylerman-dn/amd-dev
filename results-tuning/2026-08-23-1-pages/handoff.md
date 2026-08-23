# Tomorrow's live batch — ready-to-run

## 1. Book (human), e.g. {1,5,8} if free
    salloc --no-shell -N3 -w amd-mi355x-1,amd-mi355x-5,amd-mi355x-8 --gres=gpu:8 -t 720 -J ylerman-ab
    squeue -u dn -n ylerman-ab   # note JOBID

## 2. A/B-validate the 12 unvalidated scales (all_reduce first)
    cd /home/dn/ylerman/tasks/GPU-107/amd-dev
    C=/home/dn/ylerman/tasks/GPU-107/amd-dev-optuna/results-tuning/2026-08-21-1-searchcmp
    /usr/bin/python3 tools/rccl-sweep/ab_run.py --jobid <JOBID> \
      --nodelist amd-mi355x-1,amd-mi355x-5,amd-mi355x-8 \
      --outdir /home/dn/ylerman/tasks/GPU-107/amd-dev-optuna/results-tuning/2026-08-24-1-abvalidate \
      $C/all_reduce_{1,2,3}n_grid.conf $C/all_gather_{1,2,3}n_grid.conf \
      $C/reduce_scatter_{1,2,3}n_grid.conf $C/broadcast_{1,2}n_grid.conf $C/reduce_1n_grid.conf

Add --dry-run first to eyeball the commands. A/B is self-relative (config vs
default on the same nodes), so the new node set is fine.

## 3. While it runs: watcher on the job (NODE_FAIL / timeout / stall)
Claude sets this up when launching (background monitor on sacct + progress logs).

## 4. After: rebuild pages
    /usr/bin/python3 results-tuning/2026-08-23-1-pages/build_pages.py
(predicted labels flip to validated automatically once the new .validated.csv
files are pointed at - small AB map update in build_pages.py)

## Also queued for a quiet window
- fresh-set calibrate (grid truth on one node set) - tool mode not yet written
- alltoall channel sweep (small, forced RING/SIMPLE)
- broadcast 2n small-size gate decision (user)
