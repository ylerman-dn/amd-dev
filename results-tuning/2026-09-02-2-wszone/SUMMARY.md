# wszone — unit-corrected candidate sweep, all_reduce 1-node ≥64 MiB (WarpSpeed zone)

**Date:** 2026-09-02, 15:16–18:07 UTC · **Node:** amd-mi355x-des2-2 (TEST, held via alloc 20953, `ylerman-wszone`) ·
**Image:** lmsysorg/sglang:v0.5.17-rocm720-mi35x · **Plugin:** frozen `librccl-tunerv4-dn.so` (NFS `infer-2026-08-30/`) ·
**Binary:** `/opt/shared/ylerman/GPU-107/bin/all_reduce_perf -b 64M -e 512M -f 2 -g 8 -n 20 -w 5 -c 1 -A 1`

**Design:** 31 arms = no-plugin default + {ring+simple, ring+ll, tree+ll} × channels {8,16,32,56,112,168,224,256,280,320},
rule zone `allreduce,67108864,536870912,<algo>,<proto>,<ch>,1,8,-1,-1` (`sw_ws_*.conf`), 5 repeats interleaved
round-robin. Raw logs (stdout + gzipped NCCL INFO): node `des2-2:/data/ylerman/wszone-2026-09-02/`.
Verification per run: DN-TUNER "Applied config" at all 4 sizes with the right combo, 0 combo mismatches
(`verify.csv`); complete rule runs show 393 Applied lines per size. Zero-Applied runs = crashed/hung runs only.

## Verdict

**No candidate beats the default anywhere. Ship NO rule for allreduce 1-node ≥64 MiB.**
The RCCL default (RING/SIMPLE, 112 effective channels on this stack) is already optimal at all four sizes;
`ring,simple,112` reproduces it to within −0.25 % (inside rep spread), everything else loses 13–95 %.
If the conf format ever needs explicit coverage of this zone, the parity-safe line is:

```
allreduce,67108864,536870912,ring,simple,112,1,8,-1,-1
```

## Verdict table — median out-of-place busbw (GB/s) across reps, Δ vs default

| arm | n | 64M | 128M | 256M | 512M | 64M Δ% | 128M Δ% | 256M Δ% | 512M Δ% |
|---|---|---|---|---|---|---|---|---|---|
| default (no plugin) | 5 | 340.0 | 359.7 | 369.2 | 373.5 | — | — | — | — |
| ring_simple_8 | 5 | 44.3 | 46.2 | 46.4 | 46.5 | -87.0 | -87.2 | -87.4 | -87.5 |
| ring_simple_16 | 5 | 87.5 | 91.6 | 96.0 | 95.7 | -74.3 | -74.5 | -74.0 | -74.4 |
| ring_simple_32 | 5 | 164.6 | 175.0 | 182.7 | 184.8 | -51.6 | -51.3 | -50.5 | -50.5 |
| ring_simple_56 | 5 | 260.7 | 282.5 | 297.8 | 305.4 | -23.3 | -21.4 | -19.4 | -18.2 |
| **ring_simple_112** | 5 | **339.3** | **359.2** | **369.0** | **373.8** | **-0.2** | **-0.1** | **-0.1** | **+0.1** |
| ring_simple_168 | 5 | 216.4 | 300.6 | 368.9 | 373.0 | -36.4 | -16.4 | -0.1 | -0.1 |
| ring_simple_224 | 5 | 216.2 | 300.8 | 368.7 | 372.8 | -36.4 | -16.4 | -0.1 | -0.2 |
| ring_simple_256 | 0 | — | — | — | — | SIGFPE crash, 5/5 reps | | | |
| ring_simple_280 | 4 | 127.4 | 134.8 | 140.2 | 140.3 | -62.5 | -62.5 | -62.0 | -62.4 |
| ring_simple_320 | 5 | 286.4 | 306.3 | 318.9 | 325.6 | -15.8 | -14.8 | -13.6 | -12.8 |
| ring_ll_8 | 5 | 18.2 | 18.2 | 18.2 | 18.2 | -94.6 | -94.9 | -95.1 | -95.1 |
| ring_ll_16 | 5 | 35.8 | 35.6 | 35.5 | 35.6 | -89.5 | -90.1 | -90.4 | -90.5 |
| ring_ll_32 | 5 | 69.5 | 69.7 | 69.7 | 69.8 | -79.6 | -80.6 | -81.1 | -81.3 |
| ring_ll_56 | 5 | 115.2 | 116.8 | 117.6 | 117.8 | -66.1 | -67.5 | -68.2 | -68.5 |
| ring_ll_112 | 5 | 178.4 | 181.0 | 183.9 | 185.9 | -47.5 | -49.7 | -50.2 | -50.2 |
| ring_ll_168 | 5 | 177.7 | 180.8 | 183.3 | 185.1 | -47.8 | -49.7 | -50.3 | -50.4 |
| ring_ll_224 | 4 | 177.3 | 181.0 | 183.3 | 185.1 | -47.9 | -49.7 | -50.4 | -50.4 |
| ring_ll_256 | 0 | — | — | — | — | SIGFPE crash, 5/5 reps | | | |
| ring_ll_280 | 5 | 52.8 | 52.7 | 52.6 | 52.6 | -84.5 | -85.4 | -85.8 | -85.9 |
| ring_ll_320 | 5 | 130.6 | 131.3 | 131.6 | 132.3 | -61.6 | -63.5 | -64.4 | -64.6 |
| tree_ll_8 | 5 | 16.8 | 16.9 | 16.9 | 16.9 | -95.1 | -95.3 | -95.4 | -95.5 |
| tree_ll_16 | 5 | 32.8 | 32.9 | 33.1 | 33.3 | -90.3 | -90.9 | -91.0 | -91.1 |
| tree_ll_32 | 5 | 63.6 | 64.1 | 64.4 | 64.7 | -81.3 | -82.2 | -82.6 | -82.7 |
| tree_ll_56 | 5 | 105.7 | 107.5 | 108.1 | 109.1 | -68.9 | -70.1 | -70.7 | -70.8 |
| tree_ll_112 | 5 | 154.7 | 158.1 | 161.1 | 163.8 | -54.5 | -56.0 | -56.4 | -56.1 |
| tree_ll_168 | 4 | 153.6 | 156.4 | 159.9 | 162.7 | -54.8 | -56.5 | -56.7 | -56.5 |
| tree_ll_224 | 3 | 153.6 | 156.7 | 160.2 | 162.9 | -54.8 | -56.4 | -56.6 | -56.4 |
| tree_ll_256 | 0 | — | — | — | — | SIGFPE crash, 5/5 reps | | | |
| tree_ll_280 | 5 | 48.6 | 48.9 | 49.0 | 49.2 | -85.7 | -86.4 | -86.7 | -86.8 |
| tree_ll_320 | 5 | 118.8 | 120.5 | 121.3 | 122.5 | -65.0 | -66.5 | -67.1 | -67.2 |

Win criterion: >+5 % beyond max rep spread. No arm qualified at any size; default rep spread is tiny
(64M: 339.6–342.1; 512M: 373.5–373.9). Full per-rep min/max in `analysis.txt`.

## Channel-unit findings (supersedes the "×4 / 224-actual" model on this stack)

1. **Default at ≥64 MiB 1-node = RING/SIMPLE, 112 effective channels** — three independent witnesses agree:
   the `-A 1` nchannels column prints 112, the INFO `nBytes:<n> nc:112` pre/post-adjustment lines print 112
   at all 4 sizes, and the `ring,simple,112` rule ties default exactly. The INSIGHTS.md §11 model
   (56 base × 4 = 224 actual; write 224 into rules) **does not describe this stack**: 224-channel rules lose
   −36 % @64M and −16 % @128M. The correct rule unit here is **112**.
2. **The rule channel value passes through an 8-bit field.** rccl-tests' nchannels column renders it signed
   (168 → −88, 224 → −32, 280 → 24, 320 → 64) and RCCL uses it mod 256:
   - **256 → 0 → deterministic SIGFPE** ("Integer divide-by-zero" inside librccl.so, backtrace in
     `*_256_r*_stdout.log`), 15/15 runs across all three combos.
   - 280 → effective 24 channels, 320 → effective 64 channels — their busbw sits exactly on the measured
     8/16/32/56/112 channel-scaling curve (280: 127 GB/s ≈ 24ch; 320: 286 GB/s ≈ 64ch).
3. **168 and 224 clamp to a common effective count** (curves byte-identical: 216 / 300 / 369 / 373):
   over-provisioning above 112 costs −36 % @64M and −16 % @128M and washes out at ≥256M. The clamp sits
   above 112 and below the values requested; the INFO `nc:` lines cannot resolve it (they print 112 for
   every arm, including 8-channel runs — on this stack those lines do not reflect the tuner override at
   all; busbw scaling is the only reliable actual-channel witness).
4. **High/wrapped channel requests are hang-prone at init**: 5/155 runs hung before the first size row and
   were killed by watchdog/timeout — ring_ll_224_r2, tree_ll_168_r3, tree_ll_224_r3, tree_ll_224_r4,
   ring_simple_280_r5. All completed reps of those arms are self-consistent, but ≥168-channel LL rules are
   flaky as well as slow. Nothing ≤112 ever crashed or hung.

## Does forcing ring/simple lose the RING* WarpSpeed path?

**No.** `ring,simple,112` matches the no-plugin default at all four sizes within −0.25 % (inside rep
spread), so routing through the plugin with the matching combo keeps full WarpSpeed-level performance.
What *does* lose the path: any LL rule (ring_ll best-case −47 %, tree_ll best-case −54 % — LL saturates
~186 GB/s vs SIMPLE's 374 GB/s at 512M) and any channel count ≠ 112.

## Failures (20/155 runs, all accounted)

- 15 × SIGFPE: `*_256` arms, 5/5 reps each (finding #2 above).
- 5 × init hang (rc=124/137, killed at 4–17 min): listed in finding #4. One before the watchdog existed
  cost 17 min; the sweep still completed 5 full reps in 2 h 50 m.

## Files

- This dir: `SUMMARY.md`, `wszone_analysis.txt` (full per-arm/per-size stats), `verify.csv` (per-run
  Applied-line verification), `progress.log` (per-run rc/duration), `wszone_driver.sh`, `wszone_analyze.py`.
- Node des2-2: `/data/ylerman/wszone-2026-09-02/` — 155 stdout logs + gzipped NCCL INFO logs.
- Confs: `/opt/shared/ylerman/GPU-107/infer-2026-08-30/sw_ws_<algo>_<proto>_<ch>.conf` (30 files).
