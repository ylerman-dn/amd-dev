# By hand — 2026-09-08-4-qwen
- 18:14Z `salloc --no-shell -p XAI -N1 -w amd-mi355x-5 --gres=gpu:8 -t 240 -J ylerman-qwen` -> job 21065 (jobid.txt). Release by hand at the end.
- 18:14Z B1 detect launched as ONE srun step (loop over sc1, sc2, dummy; sc3 follows when check 3 lands):
  `srun --jobid=21065 -N1 -w amd-mi355x-5 bash -c 'for arm in ...; do RM_SUBSYS=INIT,TUNING,ENV IM_DROP_FLOOR=1 IM_BASE=/data/ylerman/qwen-2026-09-08 bash .../infer_paired.sh <model> qw_<arm>can 1 <arm>:<conf>; done'`
  driver stdout -> /opt/shared/ylerman/GPU-107/qwen-2026-09-08/B1_detect_a.driver.log
- Confs staged earlier by hand into /opt/shared/ylerman/GPU-107/infer-2026-08-30/ (sc1_grid48.final.conf, sc2_grid112.final.conf, sc_dummy_ring_simple_1ch.conf; sc3 pending).
- 18:2xZ second srun step queued on node 5: waits for the dummy detect to finish, then sc3 detect (1 round, TUNING logs), then the quiet campaign: `infer_paired.sh <qwen> qw 10 sc1:... sc2:... sc3:... dummy:... none:NONE` (IM_DROP_FLOOR=1). driver log /opt/shared/ylerman/GPU-107/qwen-2026-09-08/B1b_B2.driver.log. The wait loop is the only glue not in a script.
- 20:0xZ swap check launched by hand: `infer_paired.sh <qwen> qwswapcheck 2 dummy:sc_dummy_ring_simple_1ch.conf none:NONE` with RM_SUBSYS=INIT,TUNING,ENV, to prove conf swaps happen inside one shared server (campaign logs are quiet by design).
- 20:1xZ fetched node5/ by rsync EXCLUDING the server rccl logs (2.3 GB/rank, 22 GB total) and SGLang server.log; receipts extracted as text instead (campaign_srv_receipt.txt, detect_counts.txt).
- 20:2xZ `scancel 21065` after the swap check; no ylerman containers left on node 5.
- SUMMARY.md written by me from qw_score.csv (infer_score.py) + bench logs + the two receipt files.
