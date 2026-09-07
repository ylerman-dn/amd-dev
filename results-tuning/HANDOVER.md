# HANDOVER — GPU-107 tuner value (updated 2026-09-07, end of session)

Branch: `gpu107-value-runs` (base `gpu107-cli`, keep both). Index of every run: `results-tuning/RUNLOG.md`.
Pages (http://10.10.73.168:8410/): `collective_all_reduce_v2.html` (conf of record) · `verify.html` (in-model verdicts) ·
`channels.html` · `modes_matrix.html` · `rulehits.html` · `models_status.html`.

## Settled facts
1. **Conf of record: `gpu107_1n_v4_scripted.validated.csv`** (on /opt/shared/.../infer-2026-08-30/) — derived and judged 100%
   by the tool chain (`rccl_sweep --runtime container` 44-cell grid → `optimize_metrics` → `generate_tuner_config` →
   `validate_tuner_config --runtime container --drop-env NCCL_MIN_NCHANNELS`, 7 reps/arm, BOTH arms flag-off).
   **7 rules KEPT, all P(sup)=1.00**: tree/ll 64@4K, 84@8–32K, 32@64K, 112@128K, 84@256K; ring/ll 64@512K, 96@1–4M
   (+5.9..+75.7% vs flag-off default). 8M/≥16M: default optimal, no rules.
   Raw: `results-tuning/2026-09-07-1-fullgrid/` (grid raw/, ab_v4_noflag/).
2. **In-model (serving, 1-node TP-8): per-size rules are a tie** — 50 rounds/arm × 3 modes × 3 models (qwen30b/gptoss/DeepSeek-R1):
   −0.1..−1.1%, P(sup) 0.21–0.51. Ceiling effect: all_reduce ≈ few % of step time. BUT: that verify used conf v3 (flat 112 channels);
   **the v4 conf (fewer channels = less CU theft) was NEVER tested in-model — genuinely open.**
3. **Deployment env (image 112-flag + MSCCL on) beats the tuner-visible env in-model: +1.9..+6.6% tok/s, TTFT −30% (gptoss)** —
   verified 3 models. MSCCL goes default-OFF in RCCL 2.28.3 / ROCm 7.11 → the tuner path becomes deployment default then.
4. Per-collective map (deployment): alltoall never tunable (p2p); reduce_scatter/broadcast always tuner-visible; all_gather partial,
   its Direct algo unbeatable & inexpressible. Channels: real executed spans ≠ `-A` plan (validator reads truth).
5. Training exists in-house: **Arik Gelman** ran multi-node GROK1 training on this cluster (#ai-xai-poc, 2025-12-18),
   explicitly "without any RCCL optimization" — the partner for the multi-node/training value question.

## Next steps (in value order)
a. In-model A/B of the **v4** conf (window modes d128/d256, paired driver, flag-off) — the untested low-channel hypothesis.
b. Multi-node: rccl-tests probe first (tool bare runtime supports multi-node), then Arik's training harness for the real comm-share case.
c. findings/ entries (need user approval): MSCCL verdict, flag story, v4 conf, executed-channels truth.
d. Housekeeping: old 8807-page edit still uncommitted on main checkout; broadcast rules exist only in v3 (v4 = allreduce-only).

## Process rules (paid for in blood this session)
- Any derivation = the tool CLI end-to-end, never hand steps: sweep → optimize → generate → validate.
- Spell out the exact grid + BASELINE ENV (flag on/off!) and get explicit OK before running.
- RUNLOG row at launch. P(sup)≥0.95 gates. Canary/quiet split. Verify env from ENV log lines per run.
- pkill over ssh needs the [b]racket trick. Container hangs need `docker kill`, not client timeout.
