# SUMMARY — 2026-09-14-1-bigmsg: rccl-tests all_reduce, 1 node, 128M..2G, RCCL 2.30.4

Question (ylerman, 2026-09-14): above DDA's 64 MiB ceiling, where RCCL 2.30.4 runs its classic ring/tree path and a tuner is consulted, does any algo/proto/channels combination beat RCCL's own default?

Page: `results-tuning/2026-08-23-1-pages/bigmsg_2026-09-14.html`. Data: `runs_busbw.csv` (one row per run x size, requested and executed config, busbw_ip), `all_reduce_1n/live_runs.log`, `ab_out/allreduce_1n/{validate.log,per_size_stats.csv}`, `all_reduce_1n.conf` (sent to A/B), `all_reduce_1n.final.conf` (empty result).

## Setup

- Node amd-mi355x-2, job 21245 (14:01-14:24Z), image `lmsysorg/sglang:v0.5.19-rocm10-mi35x` (RCCL 2.30.4, library dated 2026-09-10), container mode, `NCCL_MIN_NCHANNELS` removed in both arms, `-n 20 -w 5 -c 1 -A 1`, INFO log per run. Node 2 was used because it was the only idle node; campaign is single-node, no fabric involved (CLAUDE.md's node-2 note is about fabric).
- Sizes 128M, 256M, 512M, 1G, 2G (`-b 128M -e 2G -f 2`). DDA is out of the way: its gate is `totalBytes <= 64 MiB`.
- Search: `rccl_tune.py run --collectives all_reduce --scales 1 --grid 32,48,56,64,112,224 --combos RING:LL,RING:LL128,RING:SIMPLE,TREE:LL,TREE:LL128 --min-size 128M --max-size 2G --image ... --minutes 300` (passthrough options added today, commits 1a40609, d9e11ae). Racing policy: every combo at anchors 32 and 48 (3 repeats), then only combos within 15% of the leader at 56, 64, 112, 224; then 3 default runs; then A/B 9 repeats per arm, keep a size only if P(sup) >= 0.95 and gain > 2%.
- 42 forced runs + 3 default runs + A/B (9+9). Every run rc=0, 0 hangs, executed config = requested config in every forced run (LL128 included).

## Result: exact parity, nothing to tune

Median busbw GB/s (3 repeats) at 128M / 2G, executed config from `-A 1`:

| requested | executed | 128M | 2G |
|---|---|---|---|
| RCCL default | RING/SIMPLE/112 (INFO log: channel{0..111} at every size) | 375 | 395 |
| RING/SIMPLE/112 | RING/SIMPLE/112 | 370 | 393 |
| RING/SIMPLE/224 | RING/SIMPLE, -A 1 shows -128 (overflow), slower | 359 | 373 |
| RING/SIMPLE/64 | same | 320 | 337 |
| RING/SIMPLE/56 | same | 295 | 308 |
| RING/SIMPLE/48 | same | 258 | 267 |
| RING/LL128/48 | RING/LL128/48 | 206 | 212 |
| RING/SIMPLE/32 | same | 183 | 185 |
| TREE/LL128/48 | TREE/LL128/48 | 160 | 163 |
| RING/LL128/32 | RING/LL128/32 | 151 | 152 |
| TREE/LL128/32 | same | 110 | 111 |
| RING/LL/48 | same | 106 | 107 |
| TREE/LL/48 | same | 99 | 101 |
| RING/LL/32 | same | 73 | 73 |
| TREE/LL/32 | same | 67 | 68 |

A/B (`validate.log`): all five sizes DROP, gains -0.04% .. +0.03%, P(sup) 0.20 .. 0.65, both arms executed RING/SIMPLE/112, 0/9 hangs in each arm. `all_reduce_1n.final.conf` has 0 rules.

## Reading

1. The default already is the best cell in the whole grid: RING/SIMPLE/112. Requesting it explicitly changes nothing (parity to 0.04%); every other cell is slower.
2. Busbw scales almost linearly with the channel count at these sizes for every protocol; the protocol order per channel is SIMPLE > LL128 > LL, and RING > TREE. So the only lever is channels, and 112 (the node's built channel count) is the maximum that helps: 224 requested runs slower (359-373).
3. LL128 is honoured on RCCL 2.30.4 at 1 node (executed LL128, 151-212 GB/s); on 2.27.7 it was silently substituted to SIMPLE (2026-08-03-2-combomatrix). Still 20% behind SIMPLE at equal channels, so it does not matter for all_reduce at these sizes.
4. Different from the old stack: the default executes 112 channels here (channel{0..111} in every default INFO log), not the 222-224 "WarpSpeed" span seen on RCCL 2.27.7 at >= 64M (2026-08-03-4-infocap, findings/11). Whether WarpSpeed is off in 2.30.4 or simply not chosen at these sizes was not investigated; for the tuner question it does not matter, the default is the best cell either way.
5. Conclusion for GPU-107 on this stack, 1 node: below 64 MiB DDA takes all_reduce without consulting the tuner; above 64 MiB the default is already optimal in the {ring, tree} x {LL, LL128, simple} x channels space. There is no size at which a 1-node all_reduce conf can add value. Campaign B (models at 128M+) therefore has no conf to test and is skipped.

## Not measured / open

- Other collectives above 64 MiB (all_gather, reduce_scatter, all_to_all, broadcast, reduce) at 1 node: not swept here.
- WarpSpeed status in 2.30.4 (see 4).
