#!/bin/bash
# Fetch 2n/3n grid + default metrics from NFS and build merged replay datasets.
set -eu
BASE=results-tuning/2026-08-18-1-newcolls
REMOTE=/opt/shared/ylerman/GPU-107/optuna-live-2026-08-18/newcolls
for coll in broadcast reduce; do
  for n in 2 3; do
    ssh amd-mi355x-1 "for f in $REMOTE/${coll}_${n}n_grid_rep*/run_*/metrics.csv; do echo \"== \$f\"; cat \"\$f\"; done" < /dev/null > $BASE/${coll}_${n}n_grid_raw.txt
    ssh amd-mi355x-1 "for f in $REMOTE/${coll}_${n}n_def_rep*/run_*/metrics.csv; do echo \"== \$f\"; cat \"\$f\"; done" < /dev/null > $BASE/${coll}_${n}n_def_raw.txt
  done
done
echo fetched
