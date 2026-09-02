# 2026-09-02-4-chanfix-llama — unit-corrected llama prefill channel retest

**Question.** The lp8kv run (RUNLOG 2026-09-02) showed llama-3.1-8B prefill band rules
(48–144 MiB) losing −38.6%..−68.2% vs no-rule. Those candidates wrote channels 16–48 into the
conf, which at ≥64 MiB are POST-×4 totals vs the default's much larger actual count
(channel-unit trap, findings/INSIGHTS.md §11). Do unit-corrected 112/168/224-total rules now
tie or beat the RCCL default?

**Setup.** Node amd-mi355x-5 (alloc 20954), image `lmsysorg/sglang:v0.5.17-rocm720-mi35x`,
model `/data/mlperf_llama31_8b/model` (TP8), `infer_paired.sh` hot-reload paired driver,
IM_ISL=8192 IM_OSL=128 conc 32 num-prompts 128, `--disable-radix-cache` all arms.
Rules: `allreduce,50331648,150994944,ring,simple,<ch>,1,8,-1,-1` (48–144 MiB band; prefill
allreduces hit it at 134217728 and 67108864 bytes).

## Finals (quiet, fresh server, 30 paired rounds × 5 arms, done 19:37 UTC)

| arm | rule channels | n | med tok/s | vs none | mean TTFT ms | mean TPOT ms | MWU p vs none | paired med Δ tok/s | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| none | (no rule) | 30 | 1235.0 | — | 738.9 | 20.31 | — | — | — |
| current | all_reduce_1n.final.conf (≤1 MiB rules) | 30 | 1239.4 | +0.36% | 745.0 | 20.18 | 0.059 | +2.0 | 0.028 |
| rs112 | 112 | 30 | 1232.6 | −0.19% | 736.0 | 20.34 | 0.96 | +0.6 | 0.61 |
| rs168 | 168 | 30 | 1175.7 | −4.80% | 827.1 | 20.90 | 1.3e-10 | −59.9 | 3.7e-09 |
| rs224 | 224 | 30 | 1179.3 | −4.51% | 821.2 | 20.84 | 2.9e-10 | −55.9 | 1.9e-09 |

## Requested vs actual channels

- The image env pins `NCCL_MIN_NCHANNELS=112`; the server comm initializes with 112 channels
  (`Channel 00/112 …`). Probe + canary detect logs show the plugin rule **applied** at both
  in-band sizes (`Applied … bytes=134217728 … channels=224` / `channels=112`), while every
  per-op line prints `nBytes:134217728 nc:112` for every arm — including none.
- The `nc:` lines are NOT a reliable actual-usage signal (peer rccl-tests sweep, RUNLOG
  2026-09-02: nc prints 112 for every arm incl. ch=8). The end-to-end deltas are the real
  clamp evidence: 168 and 224 behave identically (−4.8%/−4.5%, TTFT +12%/+11%) and both differ
  sharply from 112 — requesting more channels than the comm has is not a free clamp, it
  actively degrades prefill.

## Verdict — unit fix confirmed, correct unit is 112, no upside

1. **The −38.6%..−68.2% catastrophe is fully explained and closed.** rs112 = exact statistical
   tie with no-rule (MWU p=0.96, paired Δ +0.6 tok/s). The old losses were pure channel
   under-provisioning (16–48 vs the default's 112).
2. **The "×4 ⇒ write 224" model from INSIGHTS §11 is wrong on this stack.** On
   v0.5.17-rocm720 the parity value equals the comm channel count 112 (set by the image's
   `NCCL_MIN_NCHANNELS=112`). Writing 168/224 costs ~4.5–4.8% tok/s and +11–12% TTFT even
   though logs show nc:112. Cap any swept/corrected channel value at the comm's actual count.
3. **No rule beats default.** Best case is parity (rs112); `current` conf is in-band-inert and
   ties (+0.36%, MWU p=0.059 — marginal, ~2 tok/s). Tuner upside in the llama prefill band on
   1 node: none. Parity-safe coverage line if ever needed:
   `allreduce,50331648,150994944,ring,simple,112,1,8,-1,-1`.

## Evidence trail

- Probe lcfprobe (detect, 1 round, ring_simple_224 + none, 18:16–18:19 UTC): rule applied
  in-band (31,200 Applied @134 MB, 4,680 @67 MB, ring/simple/224), uncached prefill
  (`#cached-token: 0` ×139, no nonzero), server comm 112 channels.
- Canary lcfcan (detect, 1 round, rs112, 19:38–19:41 UTC): 31,200 Applied @134 MB + 4,680
  @67 MB, all ring/simple/112; `#cached-token: 0` ×70; nc:112 throughout.
- Finals: 150/150 reps OK, no failures. Node artifacts:
  `/data/ylerman/models-2026-08-30/{lcfprobe_*,lcf_*,lcfcan_*,paired_lcf*.log,lcf*_driver.log}`
  (all rccl detect/server logs gzipped).
- Local: `analyze.py`, `analysis.txt`, `sw_lcf_ring_simple_{112,168,224}.conf` (this dir);
  bench logs mirrored at /tmp/lcf-bench on the dev VM.

## Caveats / incidents

- GPUs were held 14:53–18:15 UTC by a foreign out-of-SLURM container (`eager_benz`, wszone
  sweep's `all_reduce_perf` deadlocked on the ring_simple_32 arm, SDMA frozen, 100% CU spin);
  waited per instructions until its owner killed it, then ran immediately. Runs themselves
  were clean and uncontended.
- Probe detect-mode tok/s (1155 vs 1222.7) is not comparable to quiet finals (asymmetric
  logging overhead on the rule arm) and was used only for rule-application/clamp evidence.
- MWU p-values are scipy asymptotic (same method as prior runs).
