# SUMMARY — 2026-09-10-1-inmodel: Qwen3-30B-A3B and gpt-oss-120b, 4 traffic modes, 4 arms, CUDA graphs ON, radix cache OFF, one server per arm, two passes

Launched 2026-09-09 22:40 Israel on 4 nodes (jobs 21122-21125), all done by 01:04 (node 8 restarted once: model missing there, fast-copied). 16 score tables, 16 hit-maps. Page: results-tuning/2026-08-23-1-pages/inmodel_2026-09-10.html

## Medians (tok/s), pass 1 / pass 2; decode all_reduce size of the mode and the sc2 rule that covers it

| model | mode | decode bytes | sc2 rule | stock | sc2 | none | dummy | sc2 vs none | stock vs sc2 |
|---|---|---|---|---|---|---|---|---|---|
| qwen | d32 | 131,072 | 128K..256K tree/ll/64 | 5169 / 5165 | 4168 / 4179 | 3514 / 3530 | 1883 / 1882 | +18.6% / +18.4% | +24.0% / +23.6% |
| qwen | p8k | 131,072 | 128K..256K tree/ll/64 | 1589 / 1588 | 1451 / 1450 | 1366 / 1365 | 117 / 117 | +6.2% / +6.2% | +9.5% / +9.5% |
| qwen | d128 | 524,288 | 512K..1024K ring/ll/64 | 16336 / 16387 | 13239 / 13252 | 10950 / 10946 | 3021 / 3023 | +20.9% / +21.1% | +23.4% / +23.7% |
| qwen | mix | 262,144 | 128K..256K tree/ll/64 | 7793 / 7795 | 6393 / 6376 | 5732 / 5727 | 1275 / 1275 | +11.5% / +11.3% | +21.9% / +22.2% |
| gptoss | d32 | 184,320 | 128K..256K tree/ll/64 | 7251 / 7291 | 5713 / 5679 | 4832 / 4820 | 2116 / 2119 | +18.2% / +17.8% | +26.9% / +28.4% |
| gptoss | p8k | 184,320 | 128K..256K tree/ll/64 | 1778 / 1777 | 1641 / 1639 | 1557 / 1559 | 112 / 112 | +5.4% / +5.2% | +8.3% / +8.4% |
| gptoss | d128 | 737,280 | 512K..1024K ring/ll/64 | 19936 / 19492 | 16761 / 16775 | 13140 / 13141 | 3029 / 3031 | +27.6% / +27.7% | +18.9% / +16.2% |
| gptoss | mix | 368,640 | none (gap) | 10033 / 10032 | 7518 / 7515 | 7518 / 7520 | 1272 / 1272 | +0.0% / -0.1% | +33.5% / +33.5% |

Source: <model>/<mode>/<pass>/score.csv (infer_score.py, ref = stock, rep 1 of each server dropped, 5 scored reps).

## Reading

1. **Ordering is identical in all 16 tables: stock > sc2 > none >> dummy.** Stock SGLang (custom all-reduce on) stays 9..24% ahead of the best RCCL-path arm in every mode; the RCCL tuner has no lever on stock single-node serving (2026-09-09-7: stock+plugin = 0 rule hits).
2. **Within the RCCL path the sc2 conf is worth +5..+28% over plain RCCL**, largest where the mode's decode all_reduce lands on a strong rule (d128: 512K-1M ring,ll,64 -> +21% Qwen, +28% gpt-oss), smallest on prefill-heavy p8k (+6% / +5%: decode is a small share and the 47-94 MB prefill chunks have no rule).
3. **Rule coverage decides**: gpt-oss mix decode all_reduce = 64 x 2880 x 2 = 368,640 B falls in the sc2 gap between 256K and 512K -> sc2 == none (0.0%). Qwen mix = 262,144 B is covered (128K-256K) -> +11.5%. A conf must cover the sizes the deployment actually produces; the sanity-check A/B used exact sizes 4K..512M x2 and left 256K..512K uncovered.
4. **dummy (ring,simple,1) costs -64..-93%** everywhere: the plugin path works, and a bad rule is catastrophic.
5. **Passes agree**: 31 of 32 arm pairs within 0.6%; the one exception is gpt-oss d128 stock (-2.2%, two transient dips in pass 2). Drift is not a factor at this effect size. Caveat on the scorer: rep 1 is the low rep in only 37 of 64 servers (on node 8 rep 2 was the low one, -5..-6%); medians survive single dips, so the tables stand, but 'drop rep 1' is a convention, not a detected cold rep.
6. TTFT: stock ~3% lower than sc2/none in every mode; dummy 10-20x. **Closing check (stockdetect/ under qwen/p8k, qwen/mix, gptoss/p8k, gptoss/mix; stock server + sc2 plugin, TUNING logs, 1 rep, node des2-2):** 0 rule hits and exactly one 4-byte all_reduce seen by RCCL in every one of the four servers. In stock SGLang the custom all-reduce takes every model all_reduce, including the 47-94 MB prefill chunks. The AGREED 'risk' item is closed: no fallback to RCCL at large sizes.
7. Hidden sizes 2048 / 2880 are corroborated by the hit-maps (every all_reduce size a multiple of 4096 / 5760 B; per-rank counts match 48 / 36 layers), not read from config.json.

## Where a tuner conf can still matter (unchanged)

Multi-node serving (custom all-reduce is intra-node), training, other collectives (reduce_scatter/broadcast are tuner-visible), frameworks that use RCCL for all_reduce. Within the RCCL path, coverage of the deployment's actual sizes (256K..512K gap) is the first fix.

## Hand steps
See HANDS.md (launch, node-8 model fast-copy + relaunch, fetches, scoring, page). Timeline: TIMELINE.md (Israel time).
