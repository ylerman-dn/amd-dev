# Run methodology — MANDATORY for every session (established 2026-09-07)

Nothing below is optional. A run that violates this protocol is invalid regardless of its numbers.

## 1. Plan-before-run (the confirmation gate)
Before ANY measurement campaign, write `results-tuning/<date>-<N>-<name>/PLAN.md` containing:
- Goal (one sentence) and the EXACT command line(s) that will run, copy-pastable.
- The full grid: every dimension and every value (no "etc", no ranges left implicit).
- Environment truth-table: NCCL_MIN_NCHANNELS present/REMOVED, RCCL_MSCCL_ENABLE value,
  runtime (container/bare), baseline arm definition — stated per arm.
- Repeats, node(s), ETA, output paths.
Paste PLAN.md verbatim in chat. **Launch only after the user replies with an explicit approval
of that plan.** Silence, questions, or "sounds interesting" are not approval.

## 2. Scripts only
Measurements launch ONLY through committed scripts in `tools/rccl-sweep/`
(rccl_sweep.py → optimize_metrics.py → generate_tuner_config.py → validate_tuner_config.py for
derivation; infer_*.sh for in-model). No inline bash drivers, no one-off scripts on nodes.
A needed new script or flag is committed and shown as a diff BEFORE its first use.

## 3. Receipts per run
- The executed command line is saved in the run dir (the tool's command.txt) and quoted in the
  RUNLOG launch row (written AT LAUNCH, completed at finish).
- The run's own NCCL INFO logs (dbg_*.log per cell / per arm) are the environment
  receipt. A run whose logs contradict PLAN.md is discarded, not reinterpreted.

## 4. No made-up numbers
Every number in chat/pages cites the file it came from. If a value isn't in a file, it doesn't
exist. Selections come from executed-channel readouts (validator), never the -A plan column.

## 5. Trust boundary (2026-09-07)
All confs and sweep conclusions from before this file are DEPRECATED for decision-making
(baseline/flag inconsistencies): gpu107_1n_v2/v3/v4*. They remain as history in RUNLOG.
Any conf intended for use is re-derived from scratch under this protocol.
