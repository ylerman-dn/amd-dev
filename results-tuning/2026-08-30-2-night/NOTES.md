# 2026-08-30-2-night — overnight inference A/B campaign 2 (node 5)

User asleep; full autonomy granted 2026-08-30 ~21:20 UTC ("as much reliable data at the end").
Approved: gptoss `--page-size 64` fix test, then more models with UNCHANGED bench params
(ISL 512 / OSL 512 / conc 32 / 128 prompts, 30 reps/arm, tuned vs default,
RCCL_MSCCL_ENABLE=0, custom all-reduce off, stock container RCCL — all as campaign 1).

## Sequence
1. gptoss page-size test: `run_many.sh <gptoss> gptossps def 1 --page-size 64` — single rep.
   Campaign-1 failure: server crashed at first prefill,
   `RuntimeError: invalid argument for batch_prefill: no matching kernel found. page_size=1, num_pages=21625429` —
   evidence: node5:/data/ylerman/models-2026-08-30/gptoss120b_def_ps1fail/server.log (renamed by campaign2 step 0a).
2. `run_campaign2.sh` (this dir; deployed to node5:/opt/shared/ylerman/GPU-107/infer-2026-08-30/):
   gptoss120b (--page-size 64), qwen235b, qwen397b — 30 reps def+tun each.
   Gzips NCCL logs per arm (disk was 98%, campaign-1 tun logs 47-62G/arm).
   Restart-safe via ATTEMPTED markers.

## Allocation
- 20859 (8h) expires ~22:57 UTC; replacement 20863 (12h, ylerman-gpu107-night) PENDING on
  node 5, takes over at expiry with no gap. scancel of 20859 was blocked by permission
  classifier — handover-by-expiry instead. Docker runs survive the handover.

## Raw outputs
node5:/data/ylerman/models-2026-08-30/{gptossps_def,gptoss120b_*,qwen235b_*,qwen397b_*}/ + campaign2.log.
Small logs pulled back into this dir at the end.

## gptoss tun hits explained (04:40Z)
15,768 hits, ALL prefill all_reduces: bytes = k×5760 for k=12..22 (69120..126720), each matched by the
RANGE rule `allreduce,65536,131072,tree,ll,48` in `all_reduce_1n.final.conf`. 7/8 rules are exact
(min=max); this one is a range — so "rules are exact-size" (prior session) is false as stated, and
gptoss is not a clean control: its prefill collectives were actively re-routed and the arm still
regressed -1.7%. Decode-size 5760B matches nothing, as expected.
Evidence: node5 zgrep of /data/ylerman/models-2026-08-30/gptoss120b_tun/logs/*.log.gz (byte histogram)
+ the conf file at /opt/shared/ylerman/GPU-107/infer-2026-08-30/all_reduce_1n.final.conf.
