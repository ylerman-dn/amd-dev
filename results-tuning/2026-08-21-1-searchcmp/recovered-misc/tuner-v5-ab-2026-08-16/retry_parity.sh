#!/bin/bash
# Preflight-gated 3n parity attempts. Each attempt is run_parity.sh, which aborts
# in ~3 min if its own preflight (5 runs, 25% spread limit) fails, and otherwise
# continues into the full interleaved A/B. Failed attempts are preserved.
# Attempts spaced 30 min; stops on success or when the allocation nears expiry.
D=/opt/shared/ylerman/GPU-107/tuner-v5-ab-2026-08-16
for attempt in 3 4 5 6 7 8 9 10 11 12; do
  # stop if <50 min left on the allocation (a full A/B needs ~35)
  left=$(squeue -h -j 14470 -o %L 2>/dev/null)
  [ -z "$left" ] && { echo "ALLOC GONE $(date -u +%FT%TZ)" >> $D/RETRY_PROGRESS; exit 1; }
  h=0; case "$left" in *-*) h=100;; *:*:*) h=$(echo $left | cut -d: -f1);; esac
  m=$(echo $left | awk -F: '{print $(NF-1)}')
  total=$((10#$h*60 + 10#$m))
  if [ "$total" -lt 90 ]; then echo "LOW TIME $left $(date -u +%FT%TZ)" >> $D/RETRY_PROGRESS; exit 2; fi
  echo "attempt $attempt starting, alloc left $left, $(date -u +%FT%TZ)" >> $D/RETRY_PROGRESS
  rm -f $D/PARITY_PROGRESS
  $D/run_parity.sh
  rc_line=$(tail -1 $D/PARITY_PROGRESS 2>/dev/null)
  echo "attempt $attempt: $rc_line" >> $D/RETRY_PROGRESS
  case "$rc_line" in
    *"rc=3"*) mv $D/parity_3n $D/parity_3n_attempt${attempt}_preflightfail; sleep 1800;;
    *"rc=0"*|*"rc=1"*)
      echo "FORWARD SUCCESS attempt $attempt $(date -u +%FT%TZ)" >> $D/RETRY_PROGRESS
      $D/run_parity_swapped.sh
      echo "swapped: $(tail -1 $D/PARITY_PROGRESS)" >> $D/RETRY_PROGRESS
      echo "ALL DONE $(date -u +%FT%TZ)" >> $D/RETRY_PROGRESS
      exit 0;;
    *) echo "UNEXPECTED '$rc_line' — stopping for inspection" >> $D/RETRY_PROGRESS; exit 3;;
  esac
done
echo "EXHAUSTED $(date -u +%FT%TZ)" >> $D/RETRY_PROGRESS
