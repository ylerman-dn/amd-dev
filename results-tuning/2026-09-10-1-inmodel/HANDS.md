# By hand — 2026-09-10-1-inmodel
- 22:40 Israel (19:40Z): launch via /opt/shared/ylerman/GPU-107/inmodel-2026-09-10.launch.sh (4 srun steps on jobs 21122-21125 booked 20:5xZ for 10 h).
- Confs used are the ones staged 2026-09-09 in /opt/shared/ylerman/GPU-107/infer-2026-08-30/ (sc2_grid112.final.conf, sc_dummy_ring_simple_1ch.conf).
- 22:5x Israel (19:5xZ): node 8 chain failed at its first server: /huggingface on node 8 has no Qwen3-30B-A3B snapshot (my pre-check counted the wrong thing).
  Killed step 21122.0, removed the stale container, fast-copied the model from the idle MI350X node (rsync over FE 10.x, 61 GB, 51 s) to
  node8:/data/ylerman/models/Qwen3-30B-A3B, relaunched the chain with that path. Node 8 is ~15 min behind the other three.
