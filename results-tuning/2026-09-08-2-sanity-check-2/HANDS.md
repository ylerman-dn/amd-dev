# By hand — sanity-check-2
- Launch line (setsid nohup rccl_tune.py run ... --grid 1,2,4,8,12,16,24,32,40,48,56,64,72,80,84,96,104,112), 16:11Z on the dev VM.
- rccl_tune.log moved into this dir after creation; PLAN.md derived from check 1's by sed; HANDS.md; RUNLOG launch row.
- Anchors stay 1,8,24,48 (validated policy); only --grid changed vs check 1.
- search_raw/ fetched by rsync (metrics.csv, merged_exec.csv built on node 8 for every run, summary.csv, command.txt). SUMMARY.md written by me from those files.
