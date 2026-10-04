# By hand — qwen-cudagraph
- 10:2xZ salloc TEST amd-mi350x-ses2-1 (MI350X! the only free node; all arms on the same node so the comparison is internal) -t 240 -> job 21079; one chained srun step: 2 detect servers (TUNING) then 2 passes x 4 servers x 6 reps via infer_many.sh (IM_CUDA_GRAPH=1). infer_many.sh takes extra server args positionally, so --disable-radix-cache is passed that way (IM_EXTRA is not read by infer_many).
- 10:5xZ attempt 1 (job 21079) FAILED: the MI350X node has no Qwen3-30B-A3B snapshot (only an empty refs/ dir under /huggingface/hub), server never came up; scancel 21079, leftover container removed.
- 11:0xZ fast-copy (rsync -aL over the FE 10.x network, 1.12 GB/s, 50 s) of the snapshot from ses2-1 to mi350x-ses2-1:/data/ylerman/models/Qwen3-30B-A3B (57 GB, 25 files, config.json present). Re-booked: job 21081
21081; chain relaunched with the local path.
- chain moved to a file (/opt/shared/ylerman/GPU-107/qwen-cg-2026-09-09.chain.sh) after an ssh-quoting failure on the inline relaunch; job 21081 step started.
