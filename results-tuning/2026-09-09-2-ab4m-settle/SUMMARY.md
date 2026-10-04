# SUMMARY — 2026-09-09-2-ab4m-settle: the 4M -73% was the OVER-REQUEST (144 > 112), not "112 at 4M"

Run: `validate_tuner_config.py --config settle_4m_ring_simple_112.conf --runtime container --drop-env NCCL_MIN_NCHANNELS --repeats 9 ...`
on amd-mi355x-ses2-1 (TEST, job 21078), ~9 min, rc=1 (no rule survived = correct outcome). Files: validate.log, per_size_stats.csv here;
node ses2-1 /data/ylerman/ab4m-2026-09-09/ab/ (dbg logs); settle_4m_ring_simple_112.validated.csv next to the conf on /opt/shared.

| size | default executed | rule | plugin applied | executed | default med | config med | gain | P(sup) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 4M | RING/SIMPLE/103 | ring,simple,112 | 112 | RING/SIMPLE/103 | 128.42 | 128.87 | +0.3% | 0.74 | DROP (parity) |

Evidence lines (ab/1n_config_r1_dbg_*.log): `DN-TUNER/Plugin: Applied config ... bytes=4194304 ... channels=112` followed by
`AllReduce: 4194304 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..102}` - RCCL trimmed the plugin's 112 to 103 exactly as it
trims the env-var request, and busbw is the default's.

Reading, with yesterday's run (2026-09-08-3-sanity-check-3/ab_out/allreduce_1n/per_size_stats.csv, rule ring,simple,144 at 4M: plugin
applied 144, executed 112 (NOT trimmed to 103), 34.8 GB/s, -73%):
- A plugin channel value <= the communicator's channel count (112 here) behaves like the env path: trimmed per call, harmless.
- A plugin channel value ABOVE the communicator's count (144) was capped to the 112 built channels (yesterday's row says ch_trimmed=yes, 144->112) but NOT trimmed further to RCCL's own per-call choice (103); all 112 ran and the
  collective collapsed 4x. Alternative still open: executing 112 channels at 4M may itself be pathological and 144 was simply the only request that produced it (the env path executing 128 at 4M gave 131 GB/s, but that communicator had 128 channels built). Mechanism inside RCCL not read (source not on this VM) - the observation is two runs, same node class, same
  size, same conf format, differing only in the requested value (112 vs 144).
- Consequence for Process A: a conf must never carry a channel value above the node's default channel count (112 on these 8-GPU MI355X
  nodes). The search's env path CAN request 128 (and gets 128 built), so a >112 winner is a trap for the plugin path. Guard options:
  clamp in generate_tuner_config.py, or reject in validate_tuner_config.py's preflight. Parked for approval (tool change).
- The rccl-tests-only claim "exec 128 is worse than default at 8M-64M" (check-3 SUMMARY) is separate and stands.
