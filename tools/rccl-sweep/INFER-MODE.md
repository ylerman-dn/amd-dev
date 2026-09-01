# In-model mode — tuning rules measured on a live model (SGLang)

Three scripts turn the rccl-sweep rule format into an end-to-end model experiment. They were
built and validated in the 2026-08-30/31 campaigns (results-tuning/2026-08-30-2-night/,
2026-08-31-1-sweep/, RUNLOG rows 2026-08-30T15:07Z … 2026-09-01T00:30Z).

## Scripts

| script | job |
|---|---|
| `infer_many.sh <model-path> <name> <def\|tun> <reps> [extra sglang args]` | one arm: start one SGLang server (docker), run N benchmark reps against it, tear down. `def` = no plugin; `tun` = DN tuner plugin + conf. |
| `infer_sweep.sh <model-path> <prefix> <size_bytes> <current_conf>` | screening: 30 single-rule confs (ring/tree × ll/ll128/simple × 8/16/24/32/48 ch) at one message size, 5 quiet reps each, then ranks and runs 30-rep finals. Restart-safe (ATTEMPTED markers). |
| `infer_finals.sh "<name:def::>" "<name:tun:<conf>:>" ...` | verified finals: given arms, 3 rotated passes × 10 reps each (drift mitigation). |

Env overrides (defaults preserve the campaign setup): `IM_BASE` (per-node results dir),
`IM_CONF_DIR` (confs + plugin .so, mounted into the container as /opt/rccl/tuner),
`IM_IMAGE` (SGLang image), `IM_TOOLDIR` (where these scripts live on the shared FS),
`RM_SUBSYS` (NCCL_DEBUG_SUBSYS; default INIT,TUNING,ENV), `RM_CONF` (conf filename for tun arms).

Benchmark parameters are frozen in `infer_many.sh` (ISL 512 / OSL 512 / concurrency 32 /
128 prompts, MSCCL off, custom all-reduce off, stock container RCCL). Changing any of them
changes what every number means — agreed change only.

## The two run modes (do not mix)

- **detect** (`RM_SUBSYS=INIT,TUNING,ENV`): per-op tuner logging on. Use for 1-rep runs only:
  which sizes the model emits, which rules fire, whether a combo is honored. NEVER for timing —
  the plugin logs EVERY consultation (millions of lines mid-benchmark); this asymmetric write
  tax produced fake regressions of 1.8–2.8% (RUNLOG 2026-08-31T05:57Z).
- **quiet** (`RM_SUBSYS=INIT,ENV`): startup-only logging, both arms identical. All timing runs.

## Hard-won rules (violating these produced retracted results)

1. **Verify requested-vs-selected per combo.** The plugin silently IGNOREs some algo/proto
   pairs (this arch: ring+ll128, tree+ll128, tree+simple — log line
   `Algorithm/protocol combination [x][y] is marked as IGNORE`); such an arm runs as no-rule.
   Probe every combo class with a 1-rep detect run before trusting its timing arm
   (RUNLOG 2026-08-31T21:20Z, 21:30Z).
2. **Interleave arms.** Node speed drifts ~1% over hours; a 30-rep block per arm let a de facto
   no-rule arm "beat" the real no-rule arm by +2% (p=1e-4!). `infer_finals.sh` rotates
   3×10-rep passes for this reason.
3. **First rep after server start is a warm-up** (20–50% low). Medians absorb it; report
   STD/CV on warm-up-trimmed data (<0.9× median) and say so.
4. **Message sizes are workload-determined:** decode = concurrency × hidden × 2B;
   prefill = packed chunk tokens × hidden × 2B. Sweep the size that a 1-rep detect histogram
   shows dominant (~98% of ops at our params).

## Findings so far (details in RUNLOG / results-tuning/2026-08-30-2-night/rules.html)

- llama8b @262144B: `ring,simple,16` +2.05% vs RCCL default (p=0.0036), 30 reps, verified.
- qwen30b @131072B: `tree,ll,16` +0.49% (p=0.009).
- RCCL default (stock container library) at both sizes: tree,ll,~112ch.
- All measured on the CONTAINER'S stock RCCL — our own librccl build hangs under SGLang
  (open issue); rccl-tests-derived confs cross that library boundary.
