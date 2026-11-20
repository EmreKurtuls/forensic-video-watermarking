#!/bin/sh
set -e
B=https://download.blender.org
( cd bbb720    && curl -sf -o bbb720.zip    $B/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov.zip && unzip -oq bbb720.zip && rm bbb720.zip && echo "bbb720 done" ) &
( cd sintel720 && curl -sf -o sintel720.zip $B/durian/movies/Sintel.2010.720p.mkv.zip && unzip -oq sintel720.zip && rm sintel720.zip && echo "sintel720 done" ) &
( cd tos720    && curl -sf -o tears_of_steel_720p.mov $B/demo/movies/ToS/tears_of_steel_720p.mov && echo "tos720 done" ) &
( curl -sf -C - -o crowd_run_1080p50.y4m https://media.xiph.org/video/derf/y4m/crowd_run_1080p50.y4m && echo "crowd_run_1080p done" ) &
( curl -sf -C - -o park_joy_720p50.y4m https://media.xiph.org/video/derf/y4m/park_joy_420_720p50.y4m && echo "park_joy_720p done" ) &
wait
echo "all downloads finished"
