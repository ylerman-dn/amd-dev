# SUMMARY — 2026-09-08-4-qwen: three sanity-check confs + dummy + none in Qwen3-30B-A3B (node 5, job 21065)

**Result: all five arms tie on decode throughput, |diff| < ~1-2% (95% CI of the paired difference, n=9).** 9 rounds each (round 1 dropped as warm-up), same server, arms interleaved per round:

| arm | conf | median tok/s | vs none | P(sup) vs none | median of per-round MEAN TTFT ms | median of per-round MEAN TPOT ms |
|---|---|---|---|---|---|---|
| dummy | ring,simple,1 channel for ALL sizes (expected bad) | 824.09 | +0.13% | 0.56 | 125.7 | 38.65 |
| none | plugin loaded, zero rules = RCCL default | 823.01 | +0.00% | 0.5 | 124.9 | 38.7 |
| sc1 | sanity-check-1 final (grid 1..48, 7 rules 4K-1M) | 820.35 | -0.32% | 0.44 | 125.7 | 38.83 |
| sc2 | sanity-check-2 final (grid 1..112, 8 rules 4K-2M) | 817.97 | -0.61% | 0.22 | 125.8 | 38.94 |
| sc3 | sanity-check-3 final (grid 1..224, 8 rules 4K-2M) | 816.64 | -0.77% | 0.22 | 127.9 | 38.88 |

Source: `qw_score.csv` (= `infer_score.py --base node5 --prefix qw --ref none`), bench logs `node5/qw_<arm>/bench_round*.log`.

## Per round tok/s (`node5/qw_<arm>/bench_round<r>.log`)

| round | sc1 | sc2 | sc3 | dummy | none |
|---|---|---|---|---|---|
| 1 | 438.2 | 807.6 | 820.8 | 828.2 | 824.8 |
| 2 | 819.0 | 818.0 | 822.4 | 829.6 | 821.7 |
| 3 | 811.4 | 804.7 | 815.5 | 824.7 | 823.1 |
| 4 | 827.6 | 819.1 | 816.6 | 835.2 | 823.0 |
| 5 | 829.5 | 814.1 | 806.9 | 807.0 | 817.9 |
| 6 | 820.4 | 818.5 | 820.9 | 824.1 | 828.6 |
| 7 | 821.4 | 805.3 | 814.5 | 814.2 | 826.4 |
| 8 | 819.3 | 810.9 | 819.3 | 828.4 | 831.4 |
| 9 | 826.9 | 819.3 | 819.1 | 819.6 | 806.7 |
| 10 | 819.8 | 822.0 | 809.3 | 823.6 | 813.0 |

Round 1 of sc1 (438) is the server's cold first benchmark; it is dropped for every arm by the scorer.

## Did the rules actually run? Yes.

- Detect runs (own server per conf, TUNING logs, 1 round; `detect_counts.txt`, node 5 `qw_<arm>can_srv/logs/`): all_reduce sizes hit = 128K (1.59M calls = decode, conc 32 x hidden 2048 x 2 B), 4K (19k), 8K (6k), 2M (2.3k). sc1 -> `tree,ll,40` at 128K; sc2/sc3 -> `tree,ll,64` at 128K (executed 64, log `channel{0..63}`); dummy -> `ring,simple,1` everywhere, executed 1 channel (`channel{0..0}`).
- Swap check in ONE shared server (2 rounds, TUNING logs, dummy vs none; `node5/qwswapcheck_*`, `paired_qwswapcheck.log`): 3,242,896 `Applied config` lines, every one `ring simple 1` = exactly 2 dummy rounds' worth (2 x ~1.62M); the none rounds applied nothing; `hot-reloaded` logged at every swap (32 lines = 4 swaps x 8 ranks). tok/s dummy 810.6 vs none 815.5 in round 2 (-0.6%). So the paired driver's per-round conf swap works, and the campaign's quiet logs (no TUNING subsystem) are the only reason it shows no reload lines.
- Env receipt, campaign server (`campaign_srv_receipt.txt`): `RCCL_MSCCL_ENABLE=0`, plugin `librccl-tunerv4-dn-hotreload.so`, `NCCL_MIN_NCHANNELS` NOT set (image flag removed).

## Reading (corrected after review, REVIEW.md)

- Decode (tok/s, TPOT): no conf, not even the 1-channel dummy, moves it beyond the noise (dummy +0.13%, P 0.56; sc2/sc3 -0.6..-0.8%, sign-test p 0.18). This is a real null but UNEXPLAINED: rccl-tests says RING/SIMPLE/1 at 128K is 65 us vs default 35 us (-46%, `../2026-09-08-1-sanity-check-1/search_raw/r0012` vs `d0000`), x96 calls/rank/step = +2.9 ms ~ +7% TPOT expected; observed 38.65 vs 38.70 ms. Where the time goes (overlap? hidden by other work?) needs a designed measurement.
- Prefill (TTFT): the campaign did NOT measure it. bench_serving reuses the same prompts every round (seed 42) and the radix cache is on, so rounds 2-10 are cache hits (TTFT ~125 ms for every arm). The only real-prefill rounds are round 1 of each server: there the dummy's TTFT was 1763 ms (`node5/qw_dummycan_dummy/bench_round1.log`) and 1762 ms (`node5/qwswapcheck_dummy/bench_round1.log`) vs 145-149 ms for sc1/sc2/sc3 and 197 ms for none -> the 1-channel conf costs ~12x TTFT on real prefill. The scorer's drop-first rule removed exactly those rounds.
- So: on this workload rules cannot help decode, and a bad conf hurts prefill badly; the sc confs are safe (145-149 ms). To measure prefill properly: `--disable-radix-cache` (IM_EXTRA) or a mode from infer_modes.sh (p2k/p8k). Proposed, not run.
- Consistent with the 2026-09-07 HANDOVER verdict (per-size rules = tie in-model) for decode.
- Not a dummy-conf failure: detect and swap-check logs prove the 1-channel rule executed (3.24M hits in the swap check, none arm 0).

## Quiet mode is not quiet

- The campaign server's rccl logs are 2.3 GB per rank (30.7M lines) with `NCCL_DEBUG_SUBSYS=INIT,ENV`; not copied (22 GB). Same for every arm, so no asymmetric tax, but worth knowing.

## By hand
See HANDS.md. Scripts did the runs (infer_paired.sh) and the verdict (infer_score.py); I typed the launch lines, the wait loop between steps, the fetch, and this file.
