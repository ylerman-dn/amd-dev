# PLAN — 2026-09-09-4-qwen-cudagraph: the deployment-realistic measurement (CUDA graphs ON), one server per arm

Hypothesis for yesterday's decode null: every in-model campaign so far ran SGLang with `--disable-cuda-graph` (needed by the paired
hot-reload design: with graphs on, RCCL collectives are captured once and the tuner is consulted only at capture, so a mid-run conf swap
changes nothing). Without graphs, decode at conc 32 for a 3B-active MoE is CPU/launch-bound (TPOT 38.7 ms), so GPU-side all_reduce
time - 35 vs 65 vs ~200 us per call - is hidden. Deployment runs WITH graphs. So: measure with graphs on, one server per arm.

Node: amd-mi350x-ses2-1 (TEST, **MI350X**, the only free node at 10:2xZ) - all arms on it, comparison is within-node only; absolute numbers not comparable to the MI355X runs.
Driver: infer_many.sh (573c356: IM_CUDA_GRAPH=1, IM_DROP_FLOOR=1 added today; shared copy updated, .bak kept).
Server: TP-8, CUDA graphs ON, --disable-custom-all-reduce, RCCL_MSCCL_ENABLE=0, image NCCL_MIN_NCHANNELS removed, --disable-radix-cache,
  plugin librccl-tunerv4-dn.so on tun arms (conf via RM_CONF), NO plugin on the def arm. Mode d32 (512/512/conc 32/128 prompts).
Arms (one server each, run in this order, then the order reversed for drift control):
  cg_none  def                               (RCCL default, no plugin)
  cg_sc2   tun RM_CONF=sc2_grid112.final.conf
  cg_dummy2 tun RM_CONF=sc_dummy2_tree_ll_1ch.conf
  cg_dummy tun RM_CONF=sc_dummy_ring_simple_1ch.conf
Reps: 6 per server, 2 servers per arm (forward + reverse order) = 12 reps/arm; rep 1 of each server is the cold benchmark (scorer drops it).
Detect first: RM_SUBSYS=INIT,TUNING,ENV, 1 rep, arms sc2 and dummy2: does "Applied config" appear at all under graph capture, how many times.
Commands (one srun step):
  for a in none:def sc2:tun dummy2:tun dummy:tun dummy:tun dummy2:tun sc2:tun none:def; do
    IM_CUDA_GRAPH=1 IM_DROP_FLOOR=1 IM_EXTRA=--disable-radix-cache IM_BASE=/data/ylerman/qwen-cg-2026-09-09 RM_CONF=<conf of a> RM_SUBSYS=INIT,ENV \
      infer_many.sh <qwen> cg<pass>_<arm> <def|tun> 6; done
Score: infer_score.py --base ... --prefix cg1 --ref none_def (and cg2), compare TPOT/tok/s across arms; expected: TPOT drops a lot vs 38.7 ms
  (graphs), and dummy/dummy2 now cost measurable decode time if the hypothesis holds.
