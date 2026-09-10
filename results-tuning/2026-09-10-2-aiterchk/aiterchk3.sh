echo "== container env AITER/SGLANG/ROCM_QUICK/RCCL =="; env | grep -E "^(AITER|SGLANG|ROCM_QUICK|RCCL|NCCL)" | sort
F=/sgl-workspace/aiter/aiter/dist/device_communicators/custom_all_reduce.py
echo "== _resolve_car_max_size body =="; sed -n '60,145p' $F
P=$(python3 -c "import sglang.srt.distributed.parallel_state as p;print(p.__file__)")
echo "== parallel_state: custom AR dispatch =="; grep -n -E "should_custom_ar|prefill_support|custom_all_reduce\(|_resolve_outplace_all_reduce_method|def all_reduce|pynccl_comm.all_reduce|torch.distributed.all_reduce|chunk" $P | head -40
echo "== parallel_state: _all_reduce_out_place / in_place bodies =="
awk '/def _resolve_outplace_all_reduce_method|def _all_reduce_out_place|def _all_reduce_in_place/{p=1;c=0} p{print NR": "$0; c++} c>40{p=0}' $P | head -150
