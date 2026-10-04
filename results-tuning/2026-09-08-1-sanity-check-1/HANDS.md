# By hand (not by script) — sanity-check-1 (attempt 3, the real one)
- Launch line itself (setsid nohup rccl_tune.py ...), typed on the dev VM at 15:28Z.
- rccl_tune.log started at results-tuning/ and moved into this dir after rccl_tune created it.
- PLAN.md (from attempt 2, tool hash + paths + default-env line updated), HANDS.md, RUNLOG launch row.
- Attempts 1 and 2 (failed fast, 15:19-15:27Z) are in ../2026-09-08-1-sanity-check-1-FAILED-attempt{1,2}/.
- search_raw/ fetched by rsync (metrics.csv, merged_exec.csv, summary.csv, command.txt per run; no dbg logs). SUMMARY.md written by me from those files.
- 17:17Z search_raw/r*/merged_exec.csv (executed algo/proto/ch per search run) built on node 8 with merge_metrics --exec-from-logs and fetched, after the run (reviewer item 4.1).
