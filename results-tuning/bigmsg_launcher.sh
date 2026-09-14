#!/bin/bash
# 2026-09-14 bigmsg campaign A: wait for an idle XAI node, then run the tune loop on it (search 128M..2G + A/B).
# Detached; progress in results-tuning/2026-09-14-bigmsg.driver.log. Kill = this script + the rccl_tune python.
cd /home/dn/ylerman/tasks/GPU-107/amd-dev
LOG=results-tuning/2026-09-14-bigmsg.driver.log
echo "=== launcher start $(date -u +%FT%TZ)" >> $LOG
while true; do
  node=$(ssh -o BatchMode=yes amd-mi355x-1 'sinfo -h -p XAI -t idle -o "%N" | tr "," "\n" | grep -oE "amd-mi355x-[0-9]+" | head -1' </dev/null 2>/dev/null)
  if [ -n "$node" ]; then
    echo "=== idle node $node at $(date -u +%FT%TZ); launching rccl_tune" >> $LOG
    python3 tools/rccl-sweep/rccl_tune.py run --collectives all_reduce --scales 1 --name bigmsg \
      --grid "32,48,56,64,112,224" --combos "RING:LL,RING:LL128,RING:SIMPLE,TREE:LL,TREE:LL128" \
      --min-size 128M --max-size 2G --image lmsysorg/sglang:v0.5.19-rocm10-mi35x --minutes 300 --nodes ${node##*-} >> $LOG 2>&1
    rc=$?
    echo "=== rccl_tune exited rc=$rc at $(date -u +%FT%TZ)" >> $LOG
    if [ $rc = 0 ]; then echo "=== BIGMSG_DONE" >> $LOG; exit 0; fi
    if grep -q "salloc.*(Requested node|not available|Unable to allocate|Pending|timed out)" $LOG; then echo "=== booking lost the race, retrying" >> $LOG; sleep 60; continue; fi
    echo "=== BIGMSG_FAILED rc=$rc" >> $LOG; exit $rc
  fi
  sleep 120
done
