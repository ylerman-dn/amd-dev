#!/bin/bash
# Campaign B' launcher (runs detached on amd-mi355x-1): one model per node. Every 2 min: for each model not yet started, take
# (a) node 2 once campaign C's sequencer prints SEQ_DONE (reuse allocation 21247 if >= 3 h remain, else re-book node 2), or
# (b) any idle XAI node (salloc 12 h). Before launching: image present, model snapshot present, >= 20 GB free on /. Missing model -> NEED_COPY line, try next model.
# Marks: $B.started_<tag>; driver logs $B.<tag>.driver.log; own log $B.launcher.log. Stop = kill this script (chains keep running as srun steps).
B=/opt/shared/ylerman/GPU-107/bigmsg-inmodel-2026-09-14
CHAIN=$B.chain.sh
IMG=lmsysorg/sglang:v0.5.19-rocm10-mi35x
declare -A MP XT BT
MP[qwen]=/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/;  BT[qwen]="32768 65536";  XT[qwen]=""
MP[gptoss]=/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/; BT[gptoss]="32768 65536"; XT[gptoss]="--attention-backend triton"
MP[dsr1]=/huggingface/hub/models--amd--DeepSeek-R1-0528-MXFP4/snapshots/913fc83b2d3962dbc2682d6b97e9ef31acb4bf5a/; BT[dsr1]="16384 32768"; XT[dsr1]=""
log() { echo "$(date -u +%FT%TZ) $*" >> $B.launcher.log; }
pending() { for t in dsr1 gptoss qwen; do [ -e $B.started_$t ] || echo $t; done; }
node_ok() {  # $1 jobid $2 node $3 tag -> 0 if image+model+disk ok
  srun --overlap --jobid=$1 -N1 -w $2 -t 3 bash -c "docker image inspect $IMG >/dev/null 2>&1 || { echo NOIMAGE; exit 1; }; [ -f ${MP[$3]}config.json ] || { echo NOMODEL; exit 2; }; f=\$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ \$f -ge 20 ] || { echo NODISK \$f; exit 3; }; echo OK \$f" </dev/null 2>&1 | grep -E "^(OK|NOIMAGE|NOMODEL|NODISK)" | tail -1
}
launch() {  # $1 jobid $2 node $3 tag
  touch $B.started_$3; echo "$3 $2 $1 $(date -u +%FT%TZ)" >> $B.nodes.txt
  log "LAUNCH $3 on $2 job $1"
  setsid nohup srun --jobid=$1 -N1 -w $2 bash $CHAIN "${MP[$3]}" $3 "${BT[$3]}" "${XT[$3]}" > $B.$3.driver.log 2>&1 < /dev/null &
}
try_node() {  # $1 jobid $2 node : launch the first pending tag whose model is on the node; returns 0 if launched
  for t in $(pending); do
    r=$(node_ok $1 $2 $t)
    case "$r" in
      OK*) launch $1 $2 $t; return 0;;
      NOMODEL) log "NEED_COPY $t on $2 (model missing)";;
      *) log "SKIP $2 for $t: $r"; return 1;;
    esac
  done
  return 1
}
log "launcher start"
while [ -n "$(pending)" ]; do
  # (a) node 2 after campaign C
  if grep -q SEQ_DONE /opt/shared/ylerman/GPU-107/sc2plus-2026-09-14.seq.log 2>/dev/null && [ ! -e $B.node2_used ]; then
    j=$(squeue -h -o "%i %L" -n ylerman-bigmsgC | head -1); jid=${j%% *}; left=${j##* }
    hrs=$(echo $left | awk -F: '{if (NF==3) print $1; else if (NF==2) print 0; else print 0}' | sed 's/.*-//')
    if [ -n "$jid" ] && [ "${hrs:-0}" -ge 3 ]; then touch $B.node2_used; try_node $jid amd-mi355x-2 && continue
    else
      out=$(salloc --no-shell -p XAI -N1 -w amd-mi355x-2 --gres=gpu:8 -t 12:00:00 -J ylerman-bigmsgB 2>&1); jid=$(echo "$out" | grep -oE "allocation [0-9]+" | head -1 | awk '{print $2}')
      if [ -n "$jid" ]; then touch $B.node2_used; try_node $jid amd-mi355x-2 && continue; else log "node2 rebook failed: $(echo $out | head -c 120)"; fi
    fi
  fi
  # (b) any idle node
  for n in $(sinfo -h -p XAI -t idle -o "%N" | tr ',' '\n' | grep -oE "amd-mi355x-[0-9]+"); do
    [ -n "$(pending)" ] || break
    out=$(salloc --no-shell -p XAI -N1 -w $n --gres=gpu:8 -t 12:00:00 -J ylerman-bigmsgB 2>&1); jid=$(echo "$out" | grep -oE "allocation [0-9]+" | head -1 | awk '{print $2}')
    if [ -z "$jid" ]; then log "salloc $n failed: $(echo $out | head -c 120)"; continue; fi
    if ! try_node $jid $n; then log "releasing $n ($jid): nothing launchable"; scancel $jid; fi
  done
  sleep 120
done
log "all models launched; launcher exit"
