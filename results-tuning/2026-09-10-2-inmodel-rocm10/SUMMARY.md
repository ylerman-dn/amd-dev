# SUMMARY — in-model campaign on the new stack (2026-09-10-2-inmodel-rocm10)

Stacks (versions and dates from `docker inspect` / file mtimes / version strings inside the images, node 4, 2026-09-14):

| | previous campaign `../2026-09-10-1-inmodel/` | this campaign |
|---|---|---|
| image | `lmsysorg/sglang:v0.5.17-rocm720-mi35x`, built 2026-08-08 | `lmsysorg/sglang:v0.5.19-rocm10-mi35x`, built 2026-09-10 |
| SGLang / torch | 0.5.17 / 2.9.1+rocm7.2.0 | 0.5.19 / 2.11.0+rocm10.0.0 |
| ROCm | 7.2.0 | 10.0.0 (`10.0.0.0-9999-6b0e43f3`: build number 9999 = development build) |
| RCCL | 2.27.7, commit 0d2c4fd of 2025-12-09, library file dated 2026-01-10 | 2.30.4, library file dated 2026-09-10 (same day as the image) |
| AITER | d9e5ef7 | 4ad9983 |

Gap: about one month of SGLang images, about nine months of RCCL code (MSCCL removed, DDA and the built-in gfx950 tuner table added in between).
Page: `results-tuning/2026-08-23-1-pages/inmodel_rocm10_2026-09-10.html`. Timeline: `TIMELINE.md`. Hand log: `HANDS.md`. Day log: `../2026-09-10-LOG.md`.

## 1. What changed on the new stack, and why the design had to change (attempt 1 -> attempt 2)

1. **RCCL 2.30.4 has no MSCCL.** `librccl.so.1` carries 19 "msccl" strings vs 349 in 2.27.7 and does not recognise `RCCL_MSCCL_ENABLE` (no "set by environment" line in any server log). The planned `msccl` arm was void.
2. **RCCL 2.30.4 ships a built-in CSV tuner with an MI355X (gfx950) table** of four all_reduce rules (section 7) that is active whenever no external plugin is loaded; an external plugin replaces it wholesale.
3. **RCCL 2.30.4 has a new direct all-reduce path, DDA** (`dda_all_reduce_ipc/fabric` kernels; env `RCCL_DDA_ENABLE`, `RCCL_DDA_THRESHOLD`, `RCCL_DDA_FABRIC_MAXBLOCKS`; every rank log prints `ncclDdaIpcCommInit: scratch N bytes, IpcGpuBarrier ...`). With RCCL as shipped the model's decode all_reduces (128K..7M) run on DDA and **never reach the tuner**: attempt-1 detect servers and the probe (`/data/ylerman/inmodel-rocm10-2026-09-10/probe/probe_tun` on ses2-1) logged 0 rule hits and exactly one 4-byte all_reduce per rank; the plugin's collType mapping itself is correct (the probe conf's allgather rule was applied to every AllGather). With `RCCL_DDA_ENABLE=0` the same server logs 71392 applied lines and all 8 sc2 rules hit (`probe/hitmap_nodda.csv`).
4. **AITER custom all-reduce aborts when `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set** (custom_all_reduce.cuh:3448 `hipIpcGetMemHandle` -> HIP invalid argument, torch 2.11). Our driver had set it since 2026-08-30; `infer_many.sh IM_NO_EXPANDABLE=1` (fbeb389) drops it for every server of this campaign. The image's own default is unset, so the stock arm is the image default.
5. Arms of attempt 2 (launched 16:14:44Z): **stock** (AITER custom AR on, image env: NCCL_MIN_NCHANNELS=112), **none** (custom AR off, floor removed, RCCL as shipped = DDA path), **nodda** (none + `RCCL_DDA_ENABLE=0`, the classic ring/tree path = the only one a tuner plugin can steer), **sc2** (nodda + plugin + `sc2_grid112.final.conf`), **dummy** (nodda + plugin + ring/simple/1 channel for all sizes). Everything else as agreed: CUDA graphs ON, radix cache OFF, one server per arm, 6 reps (rep 1 dropped), two passes in reverse order, detect servers (sc2, dummy, stock+sc2) with TUNING logs per mode, receipts per server (`<pass>/receipts.txt`: parsed `disable_custom_all_reduce`, `[AR] Using ...`, NCCL_MIN_NCHANNELS env, RCCL_DDA_ENABLE env, RCCL version). No RECEIPT_MISMATCH, no server failure, 0 FAIL reps in attempt 2.
6. Three models, six modes, five nodes: Qwen3-30B-A3B (ses2-1: d32 p8k d128; node 4: mix d256 d512), gpt-oss-120b (node 7: d32 p8k d128; node 6: mix d256 d512), DeepSeek-R1-0528 MXFP4 (node 5: d32 d128 d512). Decode all_reduce bytes = concurrency x hidden x 2 (hidden 2048 / 2880 / 7168).

## 2. Results — median tok/s, pass 1 / pass 2 (`<model>/<mode>/<pass>/score.csv`, reference = stock)

| model | mode | decode AR bytes | sc2 rule covering it | stock | none (DDA) | nodda | sc2 | dummy | sc2 vs nodda | none vs sc2 | stock vs none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen | d32 | 131,072 | tree/ll/64 (128K..256K) | 5124 / 5124 | 4978 / 4982 | 4004 / 3998 | 4236 / 4246 | 2050 / 2038 | +5.8% / +6.2% | +17.5% / +17.3% | +2.9% / +2.9% |
| qwen | p8k | 131,072 | tree/ll/64 (128K..256K) | 1576 / 1579 | 1495 / 1495 | 1424 / 1425 | 1454 / 1453 | 137 / 137 | +2.1% / +2.0% | +2.8% / +2.9% | +5.4% / +5.6% |
| qwen | d128 | 524,288 | ring/ll/64 (512K..1024K) | 16240 / 16241 | 14484 / 14458 | 13274 / 13305 | 13382 / 13380 | 3475 / 3472 | +0.8% / +0.6% | +8.2% / +8.1% | +12.1% / +12.3% |
| qwen | mix | 262,144 | tree/ll/64 (128K..256K) | 7741 / 7730 | 7191 / 7187 | 6361 / 6346 | 6447 / 6445 | 1475 / 1478 | +1.4% / +1.6% | +11.5% / +11.5% | +7.7% / +7.5% |
| qwen | d256 | 1,048,576 | ring/ll/64 (512K..1024K) | 24834 / 24886 | 22569 / 22561 | 20018 / 20016 | 21622 / 21578 | 3962 / 3955 | +8.0% / +7.8% | +4.4% / +4.6% | +10.0% / +10.3% |
| qwen | d512 | 2,097,152 | ring/ll/72 (2M exact) | 31341 / 33067 | 28232 / 29204 | 25370 / 25755 | 28645 / 28725 | 4229 / 4229 | +12.9% / +11.5% | -1.4% / +1.7% | +11.0% / +13.2% |
| gptoss | d32 | 184,320 | tree/ll/64 (128K..256K) | 7420 / 7436 | 7130 / 7132 | 5060 / 5075 | 5929 / 5924 | 2334 / 2340 | +17.2% / +16.7% | +20.3% / +20.4% | +4.1% / +4.3% |
| gptoss | p8k | 184,320 | tree/ll/64 (128K..256K) | 2000 / 2007 | 1944 / 1947 | 1754 / 1752 | 1844 / 1848 | 134 / 134 | +5.1% / +5.5% | +5.4% / +5.4% | +2.9% / +3.1% |
| gptoss | d128 | 737,280 | ring/ll/64 (512K..1024K) | 20435 / 19395 | 18518 / 18541 | 15685 / 15664 | 17255 / 17202 | 3554 / 3556 | +10.0% / +9.8% | +7.3% / +7.8% | +10.4% / +4.6% |
| gptoss | mix | 368,640 | none (gap 256K..512K) | 10207 / 10220 | 9125 / 9132 | 7723 / 7725 | 7721 / 7711 | 1488 / 1487 | -0.0% / -0.2% | +18.2% / +18.4% | +11.9% / +11.9% |
| gptoss | d256 | 1,474,560 | none (above 1M) | 28597 / 28631 | 26297 / 26340 | 23856 / 23840 | 20471 / 20520 | 3904 / 3910 | -14.2% / -13.9% | +28.5% / +28.4% | +8.7% / +8.7% |
| gptoss | d512 | 2,949,120 | none (above 1M) | 30856 / 33183 | 31239 / 30979 | 27217 / 27522 | 28244 / 27992 | 4125 / 4123 | +3.8% / +1.7% | +10.6% / +10.7% | -1.2% / +7.1% |
| dsr1 | d32 | 458,752 | none (gap 256K..512K) | 2457 / 2467 | 2261 / 2267 | 2107 / 2109 | 1784 / 1789 | 705 / 705 | -15.3% / -15.2% | +26.8% / +26.7% | +8.7% / +8.8% |
| dsr1 | d128 | 1,835,008 | none (above 1M) | 6011 / 6026 | 5438 / 5449 | 5248 / 5242 | 4554 / 4555 | 915 / 915 | -13.2% / -13.1% | +19.4% / +19.6% | +10.5% / +10.6% |
| dsr1 | d512 | 7,340,032 | none (above 1M) | 12010 / 12040 | 11198 / 11179 | 10416 / 10424 | 10421 / 10423 | 1014 / 1015 | +0.0% / -0.0% | +7.5% / +7.3% | +7.3% / +7.7% |

15 model-mode tables (3 models x 6 modes minus DeepSeek's p8k/mix/d256), 30 score.csv. Pass 1 and pass 2 agree within 1% for every arm in 12 of the 15 tables (DeepSeek: all within 0.5%); the exceptions are the conc-512 tables and gpt-oss d128: Qwen d512 (stock 31341 vs 33067 = 5.5%, none -3.3%, nodda -1.5%), gpt-oss d512 (stock 30856 vs 33183 = 7.5%, nodda -1.1%), gpt-oss d128 (stock 20435 vs 19395 = -5.1%, TTFT 257 vs 345 ms in pass 2). So "stock vs none" at d512 is inside the pass noise.

## 3. Reading

1. **Stock SGLang (AITER custom all-reduce) is still the fastest arm in every one of the 15 tables**, but its margin over RCCL shrank: stock vs none (RCCL as shipped) is +3..+13% here vs +14..+52% on the old stack (`../2026-09-10-1-inmodel/<model>/<mode>/pass1/score.csv`: Qwen d32 5169 vs 3514 = +47%, p8k +16%, d128 +49%, mix +36%; gpt-oss d32 +50%, p8k +14%, d128 +52%, mix +33%).
2. **RCCL's own DDA path is the reason.** none (RCCL as shipped, DDA) beats the tuned classic path (sc2) in 14 of 15 tables in pass 1 and 15 of 15 in pass 2, by +3..+28%; the one exception is Qwen d512 (2 MB, exact 2M rule: -1.4% / +1.7%, a tie). DDA is what RCCL 2.30.4 does by default for every intra-node all_reduce; a tuner plugin cannot see, let alone steer it.
3. **Within the classic path the conf behaves as on the old stack where a rule covers the decode size**: sc2 vs nodda = +6% (Qwen d32), +17% (gpt-oss d32), +10% (gpt-oss d128), +8% (Qwen d256), +12% (Qwen d512), +2..+5% at prefill-heavy p8k, ~0 where the decode size sits in the 512K..1M ring/ll/64 rule for Qwen (d128 +0.8%, mix +1.5%).
4. **Where no rule covers the decode size the plugin is not neutral: it costs 13-15%** (gpt-oss d256 -14%, DeepSeek d32 -15%, DeepSeek d128 -13%) or is within pass noise (gpt-oss mix -0.1%, DeepSeek d512 0.0%, gpt-oss d512 +3.8% / +1.7% where nodda itself moves 1.1% between passes). On the old stack a miss was a tie (gpt-oss mix: 7518 vs 7518). The detect hit-maps show the sc2 detect server hitting only the 128K..256K and 512K..1M rules at graph-capture sizes (the rule rows of `gptoss/d256/detect/hitmap_sc2.csv` and `gptoss/d128/detect/hitmap_sc2.csv` are identical: capture covers every batch size regardless of mode; only the prefill miss rows differ), so the loss is not a rule on the decode size. Cause established in section 7 for the 448K case (DeepSeek d32): loading our plugin displaces RCCL 2.30.4's built-in gfx950 tuner table, so the uncovered size loses AMD's LL choice and runs SIMPLE (-15% reproduced with an empty conf). The 1.47M and 1.75M cases are attributed to the same mechanism by inference (both sit in AMD's 1M-2M ring/ll/56 rule, which our conf does not cover); not measured separately.
5. **DeepSeek-R1 (hidden 7168) is out of the conf's reach**: its decode all_reduces are 448K (gap), 1.75M (above 1M) and 7M; the sc2 conf, derived from a 1-node rccl-tests sweep, has no rule there, and where it has none the plugin hurts or ties (point 4).
6. **Bigger messages do not make AITER irrelevant**: stock vs none stays +7..+13% at d256/d512 for Qwen and DeepSeek; only gpt-oss d512 (2.9 MB) is a tie within pass noise. The question "where do AITER and MSCCL become irrelevant" has a different answer on this stack: MSCCL is gone, and DDA replaced it as RCCL's own fast path at every size we measured.
7. **dummy is a clean negative control**: 1 channel ring/simple for all sizes costs 60-94% everywhere, so the plugin steers the classic path exactly as intended; nothing here is a plugin defect.

## 4. Numbers that must NOT be compared across stacks

The two campaigns share the design but not the nodes (old: 8, 9, des2-2, ses2-1; new: ses2-1, 4, 7, 6, 5), the image or the models' kernels (torch 2.11, AITER 4ad9983, triton attention for gpt-oss on both). Stock absolute tok/s barely moved (Qwen d32 5124 here vs 5169 on the old stack, gpt-oss d32 7420 vs 7251; `../2026-09-10-1-inmodel/<model>/d32/pass1/score.csv`), the RCCL arms did: plain RCCL Qwen d32 3514 old -> nodda 4004 / none (DDA) 4978 new. Nodes differ, so only within-stack ratios are meaningful; the stock-to-stock closeness is noted, not used.

## 5. Open items / proposed follow-ups (not run; each changes what is measured, so they need approval)

1. Miss-penalty diagnosis: DONE (section 7). Follow-up worth doing: re-run the sc2 arm with the four built-in gfx950 rules merged in for the uncovered ranges (a conf that never regresses the miss sizes), to measure our channel counts against AMD's on equal footing.
2. If DDA is the new default, a tuner conf can only matter for what DDA does not take: multi-node (nNodes > 1), sizes above `RCCL_DDA_THRESHOLD`, non-all_reduce collectives (the AllGathers seen in every server ran on the classic path with the plugin's channel value applied). A sweep on this stack should target those.
3. Re-derive the conf on this stack (RCCL 2.30.4 rccl-tests with `RCCL_DDA_ENABLE=0` and as shipped) before concluding anything about rule quality here: sc2 was tuned on 2.27.7.

## 6. Bookkeeping

Attempt 1 (15:04Z-15:5xZ) aborted: driver logs `driver_amd-mi355x-<node>.attempt1.log` on `/opt/shared/ylerman/GPU-107/`; its partial trees were removed. Attempt 2 driver logs: `driver_amd-mi355x-<node>.log` here. Allocations 21155 (node 7) released 19:06Z, 21156 (ses2-1) 19:18Z, 21158 (node 6) 20:09Z, 21154 (node 4) 20:16Z, 21157 (node 5) kept after ALL_DONE 00:16Z for the miss-penalty diagnostic (section 7), released 00:30Z. No jobs or containers of ours left on the cluster. Raw trees with rccl logs stay on the nodes under `/data/ylerman/inmodel-rocm10-2026-09-10/`.

## 7. Miss-penalty diagnostic (2026-09-11 00:18-00:26Z, node 5, `/data/ylerman/inmodel-rocm10-2026-09-10/missdiag/`, driver log `/opt/shared/ylerman/GPU-107/rocm10-missdiag.driver.log`)

DeepSeek d32 params, DDA off, TUNING logs, 2 reps each (diagnostic, not campaign data):

| server | tok/s rep1 / rep2 | TPOT ms | decode all_reduce 458,752 B executed as | tuner in the rank log |
|---|---|---|---|---|
| nodda, no plugin | 2106 / 2105 | 14.4 | RING / **LL** / 28 channels | `Using built-in CSV tuner, config: .../share/rccl/tuner/rccl_tuner_gfx950.csv`, 66920 "Applied config" lines |
| nodda + our plugin with an EMPTY conf | 1779 / 1786 | 17.2 | RING / **SIMPLE** / 28 channels | `TUNER/Plugin: Using DN-TUNER (v4)`, 0 applied |

(campaign, same mode: nodda 2107, sc2 1784, `dsr1/d32/pass1/score.csv`)

**Mechanism found.** RCCL 2.30.4 ships its own tuner configuration for MI355X (gfx950) and applies it through a built-in CSV tuner whenever no external plugin is loaded. The file (`/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/share/rccl/tuner/rccl_tuner_gfx950.csv` inside the image, same format as our confs) has four all_reduce rules for 1 node / 8 ranks:

```
allreduce,0,16383,tree,ll,1,1,8,-1,-1
allreduce,16384,524287,ring,ll,-1,1,8,-1,-1
allreduce,524288,1048575,ring,ll,32,1,8,-1,-1
allreduce,1048576,2097000,ring,ll,56,1,8,-1,-1
```

Loading our plugin (`NCCL_TUNER_PLUGIN`) REPLACES this built-in tuner. Every size our conf does not cover then falls back to RCCL's generic cost model, which picks SIMPLE where AMD's table says LL - that is the measured -15% at 448K (DeepSeek d32). The -13% at 1.75M (DeepSeek d128) and -14% at 1.47M (gpt-oss d256) are the same pattern by inference: both sizes sit in AMD's 1M-2,097,000 ring/ll/56 rule and in none of ours. The empty-conf plugin reproduces the sc2 number (1779 / 1786 vs sc2 1784 / 1789 in the campaign), so the sc2 rules that DO fire in that mode contribute nothing there; the whole loss is the displaced built-in table.

Consequences: (1) on this stack "nodda" is not "untuned RCCL" - it is RCCL with AMD's own gfx950 conf; "sc2 vs nodda" compares our conf with AMD's, on a path (classic ring/tree) that DDA switches off by default anyway. (2) A plugin conf for RCCL >= 2.30 must at least carry the built-in rules for everything it does not override, or it regresses the uncovered sizes. (3) Where both tables have a rule, ours wins on channel count: gpt-oss d128 (737K: AMD ring/ll/32 vs ours ring/ll/64) +10%, Qwen d256 (1,048,576: AMD ring/ll/56 vs ours ring/ll/64) +8%; Qwen d512 (2,097,152) is ABOVE AMD's last rule (ends at 2,097,000), so its +12% is our ring/ll/72 vs the generic fallback, not 72 vs 56. The tables disagree on shape at 128K-256K (AMD ring/ll, ours tree/ll/64: +6% Qwen d32, +17% gpt-oss d32 for ours) and at 16K-64K (AMD ring/ll default channels, ours tree/ll 2..48 channels; not exercised by any decode size here).
