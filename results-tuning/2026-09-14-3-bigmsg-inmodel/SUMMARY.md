# SUMMARY — 2026-09-14-3-bigmsg-inmodel (campaign B'): tune the tuner from the model at >= 128 MiB

Question (ylerman): above 64 MiB neither AITER custom all-reduce (declines) nor RCCL's DDA (threshold) act, so with the image's INT8 quick-reduce switched off the prefill all_reduces of a
live SGLang server reach RCCL's classic path. Can an in-model search over algo/proto/channels find a rule that beats RCCL's default there, where rccl-tests (campaign A) could not?

Answer: **No, for all six model x size combos.** The in-model optimum is RCCL's own default (ring/simple/112). LL128 and tree cannot even be selected in the model's communicator.
Stock SGLang with its INT8 quick-reduce is 2-6% faster than any exact RCCL configuration at these sizes.

## Setup

- Image `lmsysorg/sglang:v0.5.19-rocm10-mi35x` (RCCL 2.30.4). Custom AR ON (image default), `ROCM_QUICK_REDUCE_QUANTIZATION=NONE` on the RCCL arms, CUDA graphs ON, radix cache OFF,
  `IM_NO_EXPANDABLE=1`, `NCCL_MIN_NCHANNELS` removed on the RCCL arms. Traffic ISL 8192 / OSL 128 / concurrency 32 / 96 prompts. Metric: prefill input tok/s (and mean TTFT) from bench_serving.
- Message size = prefill batch tokens x hidden x 2 B (`--chunked-prefill-size N --max-prefill-tokens N`): Qwen 32768 -> 128 MiB, 65536 -> 256 MiB; gpt-oss 32768 -> 180 MiB, 65536 -> 360 MiB;
  DeepSeek 16384 -> 224 MiB, 32768 -> 448 MiB. Detect servers (TUNING logs) confirmed RCCL saw exactly these sizes (AR_SIZES lines: e.g. 188,743,680 B x3066, 268,435,456 B x1746, 469,762,048 B x5166).
- Per combo (chain_template.sh / chain2): detect (2 arms, TUNING) -> search on ONE hot-reload server, 13 arms x 3 rounds round-robin (prefill is not graph-captured, so each conf swap acts on the next
  rep): no rule (RCCL default), ring/simple x {8,16,24,32,48,64,80,112}, ring/ll128 x {32,64,112}, tree/ll128 x 112 -> PICK by median input tok/s; A/B with fresh servers only if the best arm beats
  the default by > 2% (never triggered) -> stock (INT8 quick-reduce) reference server, 3 reps.
- One model per node as nodes freed: gpt-oss node 6 (16:08-16:52Z), Qwen node 6 (16:59-17:44Z, snapshot fast-copied 54 s), DeepSeek node 6 (17:50-19:07Z, 376 GB fast-copied in 355 s).
  Node 8 was tried for a parallel chain and abandoned (foreign process on GPU 0). Per-model timing in Israel time: TIMELINE.md.

## Results (median of 3 rounds, input tok/s; `<model>/b<batch>/score.csv`)

| model | size | RCCL default (no rule) | ring/simple/8 | /32 | /64 | /80 | /112 | LL128 & tree arms | best vs default | stock INT8 | stock vs default |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss | 180 MiB | 133,102 | 47,340 (-64%) | 103,230 (-22%) | 126,819 (-5%) | 131,108 (-1.5%) | 133,063 | 132,861..133,205 | +0.08% (rl112) | 138,301 | +3.9% |
| gpt-oss | 360 MiB | 134,918 | 47,492 (-65%) | 104,598 (-22%) | 128,306 (-5%) | 132,075 (-2%) | 134,773 | 134,806..134,993 | +0.06% (tl112) | 141,210 | +4.7% |
| Qwen | 128 MiB | 106,785 | 45,089 (-58%) | 87,861 (-18%) | 102,897 (-4%) | 105,314 (-1.4%) | 106,648 | 106,633..106,721 | -0.06% (rl64) | 109,321 | +2.4% |
| Qwen | 256 MiB | 107,654 | 46,369 (-57%) | 88,639 (-18%) | 103,838 (-4%) | 106,323 (-1.2%) | 107,591 | 107,530..107,692 | +0.04% (rl32) | 111,070 | +3.2% |
| DeepSeek | 224 MiB | 35,603 | 11,440 (-68%) | 26,619 (-25%) | 33,590 (-6%) | 34,821 (-2%) | 35,567 | 35,562..35,591 | -0.04% (rl64) | 37,155 | +4.4% |
| DeepSeek | 448 MiB | 35,768 | 11,400 (-68%) | 26,543 (-26%) | 33,689 (-6%) | 34,974 (-2%) | 35,754 | 35,723..35,769 | 0.00% (rl64) | 37,842 | +5.8% |

Rounds agree within 0.3% everywhere (score.csv has n=3 per arm). PICK gate (> 2%) never met, so no A/B was run; the stock reference is a separate fresh server (2 scored reps).

## What the verification showed (`verify_arms.sh`, node 6, gpt-oss 180 MiB, TUNING logs; HANDS.md 16:57Z)

1. Every arm executed `RING/SIMPLE channel{0..111}` for the 188,743,680 B all_reduce: 10,220 identical lines, nothing else.
2. The plugin logged 1,752 x "[ring][ll128] is marked as IGNORE" and 1,752 x "[tree][ll128] is marked as IGNORE": RCCL hands the tuner a cost table in which LL128 and tree/ll128 are
   unavailable in the SGLang communicator (plugin.c:463 applies a rule only when its algo/proto is not IGNORE). A tuner can therefore steer only ring/simple channel counts in the model.
3. The no-rule arm (valid conf that never matches, and an empty file) and rs112 gave the same executed line and the same throughput: RCCL's in-model default IS ring/simple/112.
4. The 95-105k values seen as the first batch of every fresh server are cold start (the warm-up arm running rs112 read 104.4k too), not a different default.

## Reading

1. Fewer channels never help end to end. The CU/power/HBM argument for smaller channel counts does not show up: 8 channels cost 57-68%, 80 channels still cost 1-2%. The all_reduce at
   these sizes is bandwidth-bound and the classic path with all 112 built channels is what both rccl-tests and the model want.
2. Above 64 MiB the tuner's only lever in a SGLang server is the ring/simple channel count, and the default already sits at the maximum useful value. Combined with campaign A this closes
   1-node all_reduce for the tuner on this stack: below 64 MiB AITER/DDA never consult it, above 64 MiB the default is already optimal.
3. The deployment path (INT8 quick-reduce) beats every exact RCCL configuration by 2.4-5.8% input tok/s; whether the INT8 quantisation of prefill all_reduces affects output quality was not measured.

## Bookkeeping

Chains: gptoss (node 6, original chain; its b65536 part re-run on chain2), qwen (node 6, chain2), dsr1 (node 6, chain2). The original chain's no-rule arm used an empty conf file; the plugin parses
the CSV header as a dummy rule, so that arm was valid too (HANDS.md corrections). Run counts per model: 2 detect servers + 2 search servers (39 bench runs each) + 2 stock servers (3 reps) = 6 servers,
~90 bench runs; wall time gpt-oss 44 min (19:08-19:52 Israel), Qwen 45 min (19:59-20:44), DeepSeek 76 min (20:50-22:07); 18 servers, ~270 bench runs in total. Node 6 released 19:08Z. Raw trees with rccl logs stay on node 6 under `/data/ylerman/bigmsg-inmodel-2026-09-14/`.
