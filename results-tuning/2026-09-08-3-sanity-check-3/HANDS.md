# By hand — sanity-check-3
- Launch line (setsid nohup rccl_tune.py run ... --grid 1..224, 25 values), 17:17Z on the dev VM.
- rccl_tune.log moved into this dir after creation; PLAN.md derived from check 2's by sed; HANDS.md; RUNLOG launch row.
- search_raw/ fetched by rsync (metrics.csv, merged_exec.csv built on node 8 per run, summary.csv, command.txt). SUMMARY.md by me from those files. sc3 final conf copied by scp to the shared conf dir for the Qwen phase.
