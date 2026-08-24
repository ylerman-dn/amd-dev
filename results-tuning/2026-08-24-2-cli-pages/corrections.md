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
