# 10 — `-A 1` reports a plan, not execution; the debug log is the truth

**Claim.** The `-A 1` channel column is the pre-launch plan and can differ
wildly from execution (requested 48, -A said 48, executed 1-191 depending on
size). Its algo/proto columns cannot see AMD side-kernels: Direct
reduce_scatter is called on the execution path but absent from the -A
reporting path, so it would print RING while a different kernel runs. The
executed truth for all three values is the per-call debug-log line
`... -> Algo X proto Y channel{Lo..Hi}={a..b}`.

**Evidence.** Requested-vs-executed table:
`results-tuning/2026-08-24-1-pilot/channel_verification.md` (+`chanver.py`,
same dir; source dbg logs on /opt/shared). Direct-RS asymmetry: `git show
2e42aa8:projects/rccl/src/collectives.cc` ~424 (execution) vs
`src/rccl_wrap.cc` ~371 (reporting handles all_gather only). The tool now
records executed values per run (`merge_metrics.parse_exec_log`,
per_size_stats def_exec/cfg_exec columns).

**Date.** 2026-08-24. **Scope.** This build.
**Falsified by.** An rccl-tests/-A implementation that reports launch-time
channel spans.
