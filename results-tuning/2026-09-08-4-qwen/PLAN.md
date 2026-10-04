# PLAN — 2026-09-08-4-qwen: the three sanity-check confs + a dummy + none, in Qwen3-30B-A3B (drafted 17:3xZ, conf3 name filled when check 3 lands)

Goal: measure each validated conf in a live SGLang server (TP-8, one node), against "none" (plugin
loaded, zero rules = RCCL default), with the scripts only: infer_paired.sh for the runs, infer_score.py
for the verdict. Approved by user 17:0xZ in chat ("none arm mandatory; 1 detect round then 10 quiet rounds").

Node: amd-mi355x-5 or -9 (idle, Qwen3-30B-A3B local at
  /huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/).
Allocation: salloc --no-shell -p XAI -N1 -w <node> --gres=gpu:8 -t 240 -J ylerman-qwen (by hand; released by hand at the end).
Confs (in IM_CONF_DIR=/opt/shared/ylerman/GPU-107/infer-2026-08-30/):
  sc1_grid48.final.conf   = 2026-09-08-1-sanity-check-1/all_reduce_1n.final.conf (7 rules, 4K..1M)
  sc2_grid112.final.conf  = 2026-09-08-2-sanity-check-2/all_reduce_1n.final.conf (8 rules, 4K..2M)
  sc3_grid224.final.conf  = 2026-09-08-3-sanity-check-3/all_reduce_1n.final.conf (TBD)
  sc_dummy_ring_simple_1ch.conf = allreduce,0,4294967295,ring,simple,1  (expected BAD: 1 channel everywhere)
Plugin: librccl-tunerv4-dn-hotreload.so (driver default), DN_TUNER_HOT_RELOAD=1.
Server env (driver): RCCL_MSCCL_ENABLE=0, --disable-custom-all-reduce, IM_DROP_FLOOR=1 (image NCCL_MIN_NCHANNELS=112 REMOVED,
  same baseline as the A/Bs), ISL/OSL 512/512, conc 32, 128 prompts (frozen campaign params), default attention backend.

Step B1, detect (1 round per conf, TUNING logs on, one server each; tells which rules fire and how often):
  for arm in sc1:sc1_grid48.final.conf sc2:sc2_grid112.final.conf sc3:sc3_grid224.final.conf dummy:sc_dummy_ring_simple_1ch.conf; do
    RM_SUBSYS=INIT,TUNING,ENV IM_DROP_FLOOR=1 IM_BASE=/data/ylerman/qwen-2026-09-08 \
      srun --jobid=<id> -N1 -w <node> bash -c "bash /opt/shared/ylerman/GPU-107/infer-2026-08-30/infer_paired.sh <model> qw_${arm%%:*}can 1 $arm"
  done
Step B2, quiet performance (ONE server, 5 arms interleaved per round, 10 rounds):
  IM_DROP_FLOOR=1 IM_BASE=/data/ylerman/qwen-2026-09-08 srun --jobid=<id> -N1 -w <node> bash -c \
    "bash /opt/shared/ylerman/GPU-107/infer-2026-08-30/infer_paired.sh <model> qw 10 sc1:sc1_grid48.final.conf sc2:sc2_grid112.final.conf sc3:sc3_grid224.final.conf dummy:sc_dummy_ring_simple_1ch.conf none:NONE"
Step B3, score (scripts only): infer_score.py --base /data/ylerman/qwen-2026-09-08 --prefix qw --ref none
  (drops round 1 per arm as warm-up; median tok/s, TTFT, TPOT; gain% and same-round P(sup) vs none; rule hits from the detect runs).
Outputs: node /data/ylerman/qwen-2026-09-08/qw_<arm>/bench_round*.log, qw_srv/logs/rccl.*.log, paired_qw.log, qw_score.csv;
  copied into this dir. ETA: detect 4 x ~6 min, quiet ~10 x 5 x 1.5 min + server start = ~1h45m total.
