#!/bin/sh
set -u
export PYTHONPATH=src
PY=./.venv/bin/python
LOG=data/out/logs
mkdir -p $LOG
STEPS="$*"
step() { echo "$(date '+%H:%M:%S') start $1"; $PY "$2" > "$LOG/$1.log" 2>&1; echo "$(date '+%H:%M:%S') end   $1 (exit $?)"; }
wait_for() { while [ "$(stat -f %z "$1" 2>/dev/null || echo 0)" -lt "$2" ]; do sleep 30; done; }
want() { [ -z "$STEPS" ] || echo " $STEPS " | grep -q " $1 "; }

wait_for data/raw/park_joy_720p50.y4m 691203035
[ -f data/clips/park_joy_720p.mkv ] || $PY scripts/prepare_dataset.py park_joy_720p > $LOG/prepare_park_joy.log 2>&1
for t in c_wrong_key g_histogram b_degradation e_quality f_capacity; do want $t && step $t evaluation/$t.py; done
if want a_accuracy || want d_performance; then
  wait_for data/raw/crowd_run_1080p50.y4m 1555203036
  [ -f data/clips/crowd_run_1080p.mkv ] || $PY scripts/prepare_dataset.py crowd_run_1080p > $LOG/prepare_crowd_run.log 2>&1
fi
for t in a_accuracy d_performance; do want $t && step $t evaluation/$t.py; done
want plots && step plots evaluation/plots.py
echo "$(date '+%H:%M:%S') all done"
