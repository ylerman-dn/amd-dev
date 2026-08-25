# Running notes: things needing correction (overnight 2026-08-24/25)

Kept during the autonomous overnight runs. Normal prose, one bullet per item.

- alltoall has no CLI mode yet (channels-only sweep) - skipped overnight; needs
  a small `rccl-tune` special case.
- Node 9: ssh verified 2026-08-24, un-blacklisted, but fabric never probed -
  probe before first real use (overnight plan includes this if 9 idles).
- CLAUDE.md's node-9 "no ssh key" note was stale; fixed in both worktrees.
- 2026-08-24 ~18:47Z: stage2's A/B launch ssh hung 600s, TimeoutExpired crashed
  rccl_tune, the finally-release freed allocation 20729 while the remote ab_run
  kept running against the dead jobid and burned all 9 confs with instant
  preflight-fails. Searches were unaffected (all 9 fetched). Recovered: orphan
  killed, garbage wiped, A/B relaunched on allocation 20730. Tool fixed
  (short launch timeout + process-check verification, commit on gpu107-cli).
- 2026-08-24 19:54Z: stage3a crashed in its first search - the live oracle
  treated any substituted size as fatal for the config, but all_gather 1n has
  12 of 18 sizes force-substituted (Direct kernel), so every config died and
  the zero-winners guard stopped the run (allocation released cleanly - the
  new guards worked). Fixed: substituted sizes are dropped per size, matching
  the grid loader. Relaunched.
- 2026-08-24 20:02Z: stage3a retry crashed after a SUCCESSFUL search (the
  substitution fix held): the retry reused the same remote dir, stale run_*
  dirs from attempt 1 got concatenated into the metrics parse and a repeated
  CSV header crashed the default-run reader. Fixed: unique remote dir per
  attempt + header-tolerant parsing. Allocation auto-released cleanly again.
- 4n A/B observation (2026-08-25 ~01:00Z): preflight refusal rate visibly higher
  at 4 nodes - mixed signature: single-run collapses at large sizes (up to
  8257% spread at 256M, the exporter/stall class - four nodes mean four
  unsynchronized 30s exporter scrapes, a much larger busy-window union) plus
  small-size jitter (32-73% at 8-32K). broadcast 4n abandoned after 4 attempts;
  reduce 4n burned 3. Feeds the pending preflight-design decision (option C).
- 2026-08-25 ~02:00Z: found + fixed a verdict-integrity bug - validate wrote
  .validated.csv BEFORE its own validity gate, so an rc=2 (void) run left
  verdicts on disk; the stage4n fetch shipped an all_gather_4n file whose
  verdicts the tool itself had declared meaningless. Quarantined
  (all_gather_4n.validated.csv.INVALID-rc2, final conf removed); validate now
  writes void-run verdicts to .VOID-rc2 only.
- 5-node stage: impossible tonight - only 4 healthy idle nodes exist while
  node 4 is fabric-broken and nodes 2/3/6/7 carry co-tenants. Noted, skipped.
