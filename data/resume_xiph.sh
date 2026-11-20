#!/bin/sh
get() {  # url file size
  while [ "$(stat -f %z "$2" 2>/dev/null || echo 0)" -lt "$3" ]; do
    curl -s -C - --speed-limit 20000 --speed-time 60 -o "$2" "$1" || sleep 5
  done
  echo "$2 complete"
}
get https://media.xiph.org/video/derf/y4m/park_joy_420_720p50.y4m park_joy_720p50.y4m 691203035 &
get https://media.xiph.org/video/derf/y4m/crowd_run_1080p50.y4m crowd_run_1080p50.y4m 1555203036 &
wait
