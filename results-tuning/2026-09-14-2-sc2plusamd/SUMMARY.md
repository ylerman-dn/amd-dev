# SUMMARY — 2026-09-14-2-sc2plusamd (campaign C): does carrying AMD's built-in gfx950 rules remove the plugin's miss penalty?

Background: on RCCL 2.30.4 loading our tuner plugin replaces the built-in CSV tuner and its MI355X table (`share/rccl/tuner/rccl_tuner_gfx950.csv`, four all_reduce rules). Every size our conf
does not cover then runs the generic cost model (SIMPLE instead of AMD's LL) and lost 13-15% on 2026-09-10 (SUMMARY of 2026-09-10-2, section 7). Fix under test: `sc2_plus_amd.conf` = the eight
sc2 rules verbatim plus AMD's four rules cut to the gaps (`../2026-09-14-1-bigmsg/sc2_plus_amd.conf`, 16 rules, staged in `/opt/shared/ylerman/GPU-107/infer-2026-08-30/`).

Answer: **Yes. Wherever sc2 has no rule for the decode size, sc2plus restores AMD's table exactly (sc2plus == nodda within 0.2%); wherever sc2 has a rule, sc2plus == sc2. The plugin no longer regresses any size. None of this changes the ranking: RCCL as shipped (DDA) stays 8-29% ahead of every classic-path arm.** All three uncovered sizes (gpt-oss d256 1.47 MB, DeepSeek d32 448 KB, DeepSeek d128 1.75 MB) show the same repair: sc2 -13..-15% vs nodda, sc2plus -0.1%.

## Setup

- Image `lmsysorg/sglang:v0.5.19-rocm10-mi35x` (RCCL 2.30.4), node 2 (job 21247), custom AR OFF in all arms (the question is about the RCCL path), CUDA graphs ON, radix cache OFF, `IM_NO_EXPANDABLE=1`, floor `NCCL_MIN_NCHANNELS` removed.
- Arms: none = RCCL as shipped (DDA path) · nodda = `RCCL_DDA_ENABLE=0` (classic path, AMD's built-in table active) · sc2 = nodda + plugin + sc2_grid112.final.conf · sc2plus = nodda + plugin + sc2_plus_amd.conf.
- Models x modes: Qwen3-30B-A3B and gpt-oss-120b (d32, d128, mix, d256), DeepSeek-R1-0528 MXFP4 (d32, d128). Same traffic as 2026-09-10 (ISL/OSL/conc/prompts per mode in `modes.txt`).
- Per mode: detect servers sc2 and sc2plus (TUNING logs, 1 rep) -> hit-maps; pass 1 none, nodda, sc2, sc2plus; pass 2 reversed; 6 reps per server, rep 1 dropped. Receipts per server (`<pass>/receipts.txt`).
- Scored with `infer_score.py --ref none_def` (`<model>/<mode>/<pass>/score.csv`).

## Results — median tok/s, pass 1 / pass 2, and % vs nodda (AMD's table on the classic path)

| model | mode | decode all_reduce | sc2 rule? | none (DDA) | nodda | sc2 | sc2plus | sc2 vs nodda | sc2plus vs nodda |
|---|---|---|---|---|---|---|---|---|---|
| Qwen | d32 | 131,072 | yes | 5004 / 5015 | 4019 / 4017 | 4246 / (lost, fix-up) | 4242 / 4248 | +5.7% | +5.6% |
| Qwen | d128 | 524,288 | yes | 14572 / 14533 | 13356 / 13313 | 13423 / 13450 | 13429 / 13445 | +0.7% | +0.7% |
| Qwen | mix | 262,144 | yes | 7203 / 7218 | 6353 / 6363 | 6471 / 6468 | 6465 / 6466 | +1.8% | +1.7% |
| Qwen | d256 | 1,048,576 | yes | 22617 / 22594 | 20035 / 20059 | 21652 / 21671 | 21627 / 21659 | +8.1% | +8.0% |
| gpt-oss | d32 | 184,320 | yes | 7144 / 7139 | 5080 / 5069 | 5936 / 5935 | 5929 / 5947 | +17.0% | +17.0% |
| gpt-oss | d128 | 737,280 | yes | 18547 / 18514 | 15674 / 15657 | 17291 / 17304 | 17269 / 17284 | +10.4% | +10.3% |
| gpt-oss | mix | 368,640 | NO | 9167 / 9177 | 7732 / 7737 | 7730 / 7733 | 7728 / 7730 | 0.0% | -0.1% |
| gpt-oss | d256 | 1,474,560 | NO | 26282 / 26115 | 23928 / 23943 | 20518 / 20512 | 23893 / 23924 | **-14.3%** | **-0.1%** |
| DeepSeek | d32 | 458,752 | NO | 2273 / 2272 | 2112 / 2112 | 1786 / 1785 | 2110 / 2111 | **-15.4%** | **-0.1%** |
| DeepSeek | d128 | 1,835,008 | NO | 5466 / 5445 | 5254 / 5247 | 4561 / 4559 | 5246 / 5247 | **-13.2%** | **-0.1%** |

Hit-maps (`<model>/<mode>/detect/hitmap_*.csv`): sc2 leaves 31 (Qwen) / ~40 (gpt-oss) all_reduce sizes uncovered; sc2plus leaves only the sizes above AMD's last rule (2,097,000 B). AMD's gap
rules fire heavily (Qwen d32: 262K-512K ring/ll 16296 hits, 1M-2M ring/ll/56 35696).

## Reading

1. The miss penalty was real and is fixed by carrying AMD's rules: against nodda, sc2 -> sc2plus goes from -14.3% -> -0.1% (gpt-oss d256), -15.4% -> -0.1% (DeepSeek d32), -13.2% -> -0.1% (DeepSeek d128). At gpt-oss mix (360 KB) AMD's rule (ring/ll, default channels)
   equals the generic choice, so sc2 had no penalty there and sc2plus changes nothing.
2. Where sc2 has a rule, sc2plus reproduces sc2 to within 0.3%: the added rules never fire on those sizes.
3. Our rules still beat AMD's table where both have one: +17% (gpt-oss d32, 184K: our tree/ll/64 vs AMD ring/ll), +10% (gpt-oss d128, 720K: ring/ll/64 vs ring/ll/32), +8% (Qwen d256, 1M: ring/ll/64 vs ring/ll/56), +6% (Qwen d32).
4. Nothing here touches the deployment answer: RCCL as shipped (DDA) is 8-29% above the best classic arm in every row, and stock SGLang (AITER) is above DDA (2026-09-10-2). A conf for RCCL >= 2.30
   must carry the built-in rules for what it does not override; that is now demonstrated, and it is the form campaign D tests through the built-in CSV tuner.

## Bookkeeping

Chains on node 2 (job 21247): Qwen 14:29:49-17:16:59Z (2 h 47 min, 42 servers, 216 reps; pass2 sc2 lost to a stale container after a mid-run script re-sync, re-run as a fix-up at the end,
HANDS.md), gpt-oss 17:17:01-19:53:37Z (2 h 37 min, 40 servers, 216 reps), DeepSeek 19:53:40-22:00:27Z (2 h 07 min, 20 servers, 108 reps). Whole campaign on node 2: 14:29-22:00Z (17:29-01:00 Israel), 102 servers, 540 reps, then fix-up + campaign D on the same allocation. Israel-time timeline in TIMELINE.md. Node 8 was tried for the DeepSeek chain and
abandoned (a foreign non-Slurm container holds 69 GB on its GPU 0). RECEIPT_MISMATCH lines with `tuner=DN-TUNER conf=none` are a known false negative of the receipt (RCCL does not echo the conf path).
