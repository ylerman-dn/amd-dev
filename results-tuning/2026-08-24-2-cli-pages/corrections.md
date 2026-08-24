# Running notes: things needing correction (overnight 2026-08-24/25)

Kept during the autonomous overnight runs. Normal prose, one bullet per item.

- alltoall has no CLI mode yet (channels-only sweep) - skipped overnight; needs
  a small `rccl-tune` special case.
- Node 9: ssh verified 2026-08-24, un-blacklisted, but fabric never probed -
  probe before first real use (overnight plan includes this if 9 idles).
- CLAUDE.md's node-9 "no ssh key" note was stale; fixed in both worktrees.
