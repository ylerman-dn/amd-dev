#!/bin/bash
# User-approved 2026-08-17 ~10:20Z: stop amd-nic-metrics-exporter on nodes 5,6,7,
# run the swapped parity pass, restart the exporter afterwards NO MATTER WHAT.
D=/opt/shared/ylerman/GPU-107/tuner-v5-ab-2026-08-16
restart_exporters() {
  for n in 5 6 7; do
    ssh -o BatchMode=yes amd-mi355x-$n 'sudo -n systemctl start amd-nic-metrics-exporter; systemctl is-active amd-nic-metrics-exporter' </dev/null \
      | sed "s/^/node$n exporter: /" >> $D/EXPORTER_LOG
  done
  echo "exporters restarted $(date -u +%FT%TZ)" >> $D/EXPORTER_LOG
}
trap restart_exporters EXIT
: > $D/EXPORTER_LOG
for n in 5 6 7; do
  ssh -o BatchMode=yes amd-mi355x-$n 'sudo -n systemctl stop amd-nic-metrics-exporter; systemctl is-active amd-nic-metrics-exporter; pgrep -c -x nicctl || echo no-nicctl' </dev/null \
    | sed "s/^/node$n stop: /" >> $D/EXPORTER_LOG
done
echo "exporters stopped $(date -u +%FT%TZ)" >> $D/EXPORTER_LOG
[ -d $D/parity_3n_swapped ] && mv $D/parity_3n_swapped $D/parity_3n_swapped_s4i_preflightfail
echo "no-exporter swapped attempt s5x starting $(date -u +%FT%TZ)" >> $D/RETRY_PROGRESS
rm -f $D/PARITY_PROGRESS
$D/run_parity_swapped.sh
echo "s5x: $(tail -1 $D/PARITY_PROGRESS) (exporters restarting now)" >> $D/RETRY_PROGRESS
