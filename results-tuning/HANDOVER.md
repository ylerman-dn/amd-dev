# HANDOVER — GPU-107 tuner-value work (updated 2026-09-07)

## Where things stand
1. **Solid, verified 3× (do not re-litigate):** in-model, the deployment env (image's
   `NCCL_MIN_NCHANNELS=112` + MSCCL on) beats the tuner-visible env (flag removed, MSCCL off)
   by **+3..+6.6% tok/s** on qwen30b / gpt-oss-120b / DeepSeek-R1 at n≈49 rounds/arm.
   MSCCL becomes default-OFF in RCCL 2.28.3 (ROCm 7.11) — when the image moves there, the
   tuner-visible path becomes the deployment default and our conf work becomes directly relevant.
2. **Microbenchmark (rccl-tests, correct 8-proc container harness):** conf `gpu107_1n_v2/v3`
   beats vanilla by +22..+76% (allreduce 8K–2M) and +118..+171% (broadcast 512K–2M); ties image
   default except the RING/LL window (512K–2M: +74/+68/+18%). Detailed evidence pages below.
3. **In-model: the conf is a statistical TIE on all 3 models** (30+ mode-runs, then 50-round
   verify) — all_reduce is too small a share of serving step time at every mode tested.

## THE OPEN FLAW (why the conf is not final)
The 2026-09-06 "resweep" did NOT run the full combo × channel grid:
combos were swept only at 112 channels; channels only at default combos; the winning
RING/LL window was never channel-laddered; and in the final env (flag removed) the channel
dimension has only 2 measured points. **Next session's first job: run the real grid via the
tool** — {ring/ll, ring/simple, tree/ll} × channels {1,2,4,8,16,32,48,64,84,96,112} × sizes,
flag removed, MSCCL=0, `--runtime container` (now the tool default), 5 reps — derive conf v4
with per-size A/B gates (P(sup)≥0.95), THEN re-run the window-mode model A/Bs against v4.

## Branches / repo
- Work branch: `gpu107-value-runs` (= `gpu107-cli` + 46 commits). `gpu107-cli` = stable base;
  keep it, work only on value-runs. Tool commits that matter: e2e8985 (container runtime),
  345f0a8 (infer_modes/paired knobs), ada84ca (hot-reload plugin). Everything else = evidence
  commits indexed one-row-each in `results-tuning/RUNLOG.md` (the only index you need).
- Git tracks summaries/pages/confs (~2MB); raw `.log`/`.txt` are gitignored by design — raw
  data lives on the nodes (paths in each RUNLOG row) + `raw/` copies in some run dirs.
- `findings/` was NOT updated during these campaigns (entries need user approval; none written).

## Evidence pages (http://10.10.73.168:8410/)
`verify.html` (pending, final 50-round tables) · `modes_matrix.html` (30 mode-runs) ·
`conf_v3_sweep.html` (kept/dropped + raw reps) · `channels.html` (every channel value vs 112) ·
`rulehits.html` (rule-hit heat map) · `models_status.html` · `env224.html` · older: `rules.html`.

## Process rules for the next session (lessons paid for)
1. Before ANY derivation campaign: write the exact grid (dimensions × values × env) in one
   message and get an explicit OK — "resweep" without a spelled-out grid caused this flaw.
2. Use `tools/rccl-sweep` (rccl_sweep.py / validate_tuner_config.py, container runtime) as the
   single auditable entrypoint — no ad-hoc drivers for derivation runs.
3. RUNLOG row at LAUNCH (goal + grid), completed at finish — drift becomes visible early.
4. Canary (detect) + quiet (measure) split; P(sup)≥0.95 gates; warm-round trimming; both arms
   always same env; MSCCL/flag state verified from the ENV log lines every run.
