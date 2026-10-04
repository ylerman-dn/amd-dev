# SUMMARY — 2026-09-09-3-gptoss-modes: gpt-oss-120b, radix cache OFF, graphs off (paired driver), d32 and p8k

Node amd-mi355x-ses2-1 (TEST, MI355X), job 21078. One server per mode, arms sc2 / dummy / none interleaved per round, 6 rounds (round 1 dropped), --attention-backend triton --disable-radix-cache, image NCCL_MIN_NCHANNELS removed, MSCCL=0, custom all-reduce off. Scores node/gm_*_score.csv.

## d32 (ISL 512 / OSL 512 / conc 32 / 128 prompts)

| arm | median tok/s | vs none | P(sup) | per-round MEDIAN TTFT ms (median over rounds 2-6) | TPOT ms |
|---|---|---|---|---|---|
| dummy | 886.19 | -9.4% | 0.0 | 1883 | 32.59 |
| none | 978.38 | +0.0% | 0.5 | 154 | 32.46 |
| sc2 | 971.18 | -0.7% | 0.4 | 155 | 32.7 |

## p8k (ISL 8192 / OSL 128 / conc 32 / 96 prompts, prefill-heavy)

| arm | median tok/s | vs none | P(sup) | per-round MEDIAN TTFT ms (median over rounds 2-6) | TPOT ms |
|---|---|---|---|---|---|
| dummy | 121.2 | -82.7% | 0.0 | 17741 | 128.13 |
| none | 701.15 | +0.0% | 0.5 | 1084 | 37.65 |
| sc2 | 702.42 | +0.2% | 0.4 | 1040 | 37.84 |

Sizes in the model (detect runs, node ses2-1 gm_*can_srv/logs): decode all_reduce 184,320 B (conc 32 x hidden 2880 x 2) -> sc2 rule 128K-256K tree,ll,64; prefill chunks 2.9 MB (d32), 47 MB and 94 MB (p8k) -> no sc2 rule (rules stop at 2M; >=4M was parity in rccl-tests).

Reading: with real prefill (radix off) the 1-channel dummy costs -9% (d32) and -83% (p8k) of throughput and 12x TTFT; sc2 ties none in both modes because, graphs off, decode is CPU-bound (TPOT unchanged for every arm) and sc2 has no rule at the prefill sizes. See -5- (graphs on: sc2 +18% vs none) and -6- (stock SGLang +27% over sc2).
