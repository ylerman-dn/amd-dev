#!/bin/bash
D=/opt/shared/ylerman/GPU-107/tuner-v5-ab-2026-08-16
for attempt in s1 s2 s3 s4 s5 s6 s7 s8; do
  left=$(squeue -h -j 14470 -o %L 2>/dev/null)
  [ -z "$left" ] && { echo "ALLOC GONE $(date -u +%FT%TZ)" >> $D/RETRY_PROGRESS; exit 1; }
  echo "swapped attempt $attempt starting $(date -u +%FT%TZ)" >> $D/RETRY_PROGRESS
  rm -f $D/PARITY_PROGRESS
  $D/run_parity_swapped.sh
  rc_line=$(tail -1 $D/PARITY_PROGRESS 2>/dev/null)
  echo "swapped attempt $attempt: $rc_line" >> $D/RETRY_PROGRESS
  case "$rc_line" in
    *"rc=3"*) mv $D/parity_3n_swapped $D/parity_3n_swapped_${attempt}_preflightfail; sleep 600;;
    *"rc=0"*|*"rc=1"*) echo "SWAPPED SUCCESS $attempt $(date -u +%FT%TZ)" >> $D/RETRY_PROGRESS; exit 0;;
    *) echo "SWAPPED UNEXPECTED $rc_line" >> $D/RETRY_PROGRESS; exit 3;;
  esac
done
echo "SWAPPED EXHAUSTED $(date -u +%FT%TZ)" >> $D/RETRY_PROGRESS
