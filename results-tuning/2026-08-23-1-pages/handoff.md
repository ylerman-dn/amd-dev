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


## Status update 2026-08-23 evening (autonomous)
- DONE: live A/B of all 1n/2n scales on {5,8} (job 20713, released). 8 verdicts,
  pages updated and committed. broadcast 2n unmeasurable on a second node pair
  (18 attempts total) - needs a user gate decision, not more attempts.
- REMAINING: all_reduce/all_gather/reduce_scatter 3n - need 3 healthy nodes
  (node 4 broken). Hourly cron watches for a window and will run them.
- Rule of engagement kept: no idle allocations; 20713 cancelled at batch end.

## Morning close-out 2026-08-24 ~05:00Z (autonomous)
- No valid 3-node window appeared all night: idle sets were always {5,8} plus
  only forbidden nodes (2 orchestrator, 9 no-ssh-key, 4 fabric-broken).
- The three 3n A/Bs (all_reduce/all_gather/reduce_scatter) remain the only
  missing verdicts. One ab_run.py command runs them when 3 healthy nodes free
  up - see the command template above.
- Worth knowing: node 4 ran an 8-hour dn job overnight (20714) and sits idle
  now - it may have been repaired. A 10-minute 2n fabric probe ({1,4} or
  {5,4}, default all_reduce x3) would settle whether it can serve as the
  third node. User decision.
- Safety-net cron deleted; nothing holds any allocation.

## Preflight-noise investigation results (2026-08-24, subagent, read-only)
- 10 of 11 preflight failures = a SINGLE run dipped while the other two agreed
  within 1-5%; the dips temporally overlap the amd-nic-metrics-exporter's 30s
  scrape windows (10-11s busy each; timestamps correlated on node 5). The
  exporter also hits 1-NODE runs via host-CPU contention (~155% avg CPU) - so
  finding 05's "multi-node only" scope note is too narrow and needs a fix.
- The preflight design amplifies: 3 single runs, any 1 of 18 sizes >25% fails
  the attempt. Measured single-run dip rate 8-17% -> predicted 23-42% attempt
  failure, matching the observed 25-50%.
- broadcast 2n is genuinely different: 8K blew the limit in 14/14 attempts
  with all-3-runs scatter - chronic instability at 4-32K, not the exporter.
- USER DECISIONS needed (measurement parameters - not implemented):
  1. Preflight fix - recommended option C: when exactly one run is the outlier
     at every blown size, run a 4th; pass on 2-run consensus. Alternative B:
     5 runs, trim min+max. Option A (best-2-of-3) is unsafe - would pass the
     chronic broadcast 2n scatter.
  2. broadcast 2n gate: judge spread only over rule-covered sizes (>=32K), or
     accept it as unmeasurable.
  3. node 5 has a crash-looping node-exporter.service (324,491 restarts, port
     conflict) - report to cluster admin.

## Overnight summary 2026-08-24/25 (autonomous; all times UTC)

What ran (all through rccl-tune, per-size verdicts, 9 repeats/arm, executed-truth columns):
- stage2 (all_reduce, broadcast, reduce_scatter x 1/2/3n on {1,3,8}): 9/9
  searches; A/B 8/9 verdicts. broadcast 2n abandoned on a THIRD node pair -
  22 attempts lifetime; its instability is intrinsic.
- stage3a (all_gather, reduce x 1/2/3n on {3,5,8}): first fully clean
  end-to-end CLI run. 6/6 searches, 6/6 verdicts (all_gather 1n: defaults win,
  correct empty config).
- stage4n (all 5 collectives at 4 NODES on {1,5,8,9} - first ever): 5/5
  searches (all_reduce/broadcast/reduce/reduce_scatter 18 winners each,
  all_gather 6); A/B window hostile after midnight - only all_reduce 4n
  validated (rc=0 first try); the other four exhausted retry budgets.
- node probes: node 4 STILL broken (3/3 timeouts); node 9 HEALTHY and served
  its first workload (unblacklisted).
- 5-node stage: infeasible - never 5 healthy idle nodes.
- alltoall: skipped (no CLI channels-only mode yet).

Tool bugs found by the runs and fixed (all committed on gpu107-cli):
1. A/B launch ssh hang crashed the run and orphaned a live ab_run (fixed:
   short launch timeout + process verification).
2. Live oracle killed whole configs on any substituted size (fixed: per-size
   drop, grid semantics) - unblocked all_gather.
3. Retry over a reused remote dir concatenated stale metrics (fixed: unique
   dir per attempt + header-tolerant parsing).
4. validate wrote .validated.csv for rc=2 VOID runs (fixed: void verdicts go
   to .VOID-rc2; the one shipped file quarantined).

Pages (all dark, executed-truth columns): main site + per-run pages
(cli/2026-08-24-1-pilot, -1-stage2, -2-stage3a, -1-stage4n) + approaches.html
+ approaches-draft.html (style options for your pick).

Decisions waiting for the user:
1. broadcast 2n (and now most 4n) A/B unmeasurable under current gates -
   preflight option C recommended (see the noise investigation).
2. The 10 one-liner facts: approve/strike -> findings.
3. approaches-draft.html: pick styles per topic.
4. Node 4: report to cluster admin (probe evidence in RUNLOG).

## Session-state snapshot 2026-08-25 evening (context-compaction insurance)

State: all trees clean; no allocations; site at http://10.10.73.168:8807/
(server rooted at results-tuning/2026-08-23-1-pages, cli/ symlink inside).
Tool branch gpu107-cli (amd-dev worktree); results branch gpu107-optuna.

Data status: searches complete for 5 collectives x scales 1-5 (+ alltoall
blocked in sweep_parser, N/A -A columns). Live A/B verdicts: all 1n/2n/3n
scales (except broadcast 2n - chronic 4-16K scatter, 22 attempts, 3 pairs)
+ all_reduce 4n. 4n/5n others: searches done, A/B window-limited - option C
fixed preflight, the A/B noise gate now binds at >=4 nodes (one exporter per
node).

Agreed policies: per-size A/B verdicts, merge only in final configs; commit
discipline in CLAUDE.md; repeats 9; option C preflight (deployed); executed
values from debug logs everywhere (-A 1 channel column retired as source).

Pending user decisions: (1) approaches-draft style picks; (2) rule-scoped
preflight - judge only rule-covered sizes (broadcast 2n unlock); (3) alltoall
sweep_parser N/A fix; (4) node-4 fabric report to cluster admin.

Key paths: findings 09/10/11 written; corrections log at
results-tuning/2026-08-24-2-cli-pages/corrections.md; per-collective pages
built by results-tuning/2026-08-23-1-pages/build_site.py (auto-resolves
freshest verdicts); CLI = tools/rccl-sweep/rccl_tune.py (~/.local/bin/rccl-tune).

## 2026-08-26 session (autonomous, user away)

User approvals executed: (2) rule-scoped preflight - implemented, tested,
deployed (`--preflight-scope rules`, ab_run passes it); (3) alltoall - full
pipeline built and run at 1-5 nodes.

alltoall reality (source-verified, 2e42aa8 enqueue.cc): decomposed into p2p
send/recv tasks - NCCL_ALGO/PROTO no-ops, tuner plugin never consulted,
-A 1 prints N/A. Tool changes: N/A-tolerant sweep_parser, channels-only
search (pseudo-combo "P2P/-"), env-arm A/B (NCCL_MIN/MAX_NCHANNELS as the ON
arm, verified via the debug log's "set by environment" echo), p2p-channel
exec truth. Results: defaults win at EVERY scale (1n/3n by A/B rc=1, 2n/5n
at the search stage, 4n voided rc=2 x4 - the usual >=4n transients).
Exec-truth bonus: at 1n a 40-channel request executes the same 64 p2p
channels as the default (WarpSpeed x4); at 2n requests cap at 32; at 5n 40
rounds up to 64.

broadcast 2n UNLOCKED: first verdict after 22 refusals. Scoped preflight
excluded 4K/16K, judged the rest; 4 rules verified (32K +17.9%, 64K +104.7%,
128K +66.0%, 256K +30.6%, ring/ll128 ch=1) -> broadcast_2n.final.conf
(one merged rule 32768-262144).

5n noise analyzed offline (stage5n/NOISE-ANALYSIS.md): no 30s/60s wall-phase
clustering of outliers - wall-aligned exporter scrapes unsupported; signature
is transient fabric stalls at all sizes. Exporters-stopped A/B (admin) stays
the definitive test.

Incidents (corrections.md has details): sync_tool never pushed the
remote-executed stack (fixed, list widened); refill launcher ssh hung ~5h and
burned allocation 20783 (broadcast_2n verdict landed before that; fixed
script refill_ab2 has a 30s launch timeout).

Still open: reduce/reduce_scatter/all_gather 4n + all five 5n A/B refills -
blocked by a multi-node co-tenant (regeveyal hpo night, nodes 1,3,5,7); a
window watcher relaunches refill_ab2.sh when nodes 3,5,6,7,8 go idle with an
empty pending queue. broadcast 4n gave up again (rc=3,2,2,2).

## 2026-08-27 session close

cv-core noise gate adopted everywhere (user decision): MAD outlier
rejection, CV<=8%, core>=7, verdicts on cores; all 395 rules re-judged -
120 kept / 275 dropped; the whole 5n row gained verdicts from stored
runs (provenance-noted). Site: overview.html (gains heatmap + shipped
configs), audit.html (every accept/reject sanity-checked + 6-rule retry
shortlist), value.html (DRAFT - awaiting user review, not linked/committed),
noise-dig on :8808. Servers on :8807/:8808 are detached and survive this
session. Pending user: approaches-page review, value.html review, retry
batch for the 6 suspect drops, finding drafts approval, gpu107-optuna
worktree retirement.
