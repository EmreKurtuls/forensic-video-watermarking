#!/bin/sh
set -e
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
P="$(cd "$(dirname "$0")/.." && pwd)"
T="$(mktemp -d)"
mkdir -p "$T/wa" "$T/ns" "$P/docs"
shot() { "$CHROME" --headless=new --disable-gpu --hide-scrollbars --window-size=1280,720 --virtual-time-budget="$3" --screenshot="$2" "$1" 2>/dev/null; }

walkthrough() {
  i=0
  for f in $(awk 'BEGIN { for (x = 0; x < 480; x++) printf "%.4f ", x / 80 }'); do
    shot "file://$P/presentation/watermark_animation/index.html?freeze=$f" "$T/wa/$(printf %04d $i).png" 3000
    i=$((i + 1))
  done
}

trace() {
  i=0
  for s in 1 2 3 4 5 6 7 8; do
    for t in $(awk 'BEGIN { for (x = 1; x <= 50; x++) printf "%d ", x * 50 }'); do
      shot "file://$P/presentation/network_sim/index.html?step=$s" "$T/ns/$(printf %04d $i).png" "$t"
      i=$((i + 1))
    done
    last="$T/ns/$(printf %04d $((i - 1))).png"
    for k in $(awk 'BEGIN { for (x = 0; x < 20; x++) printf "%d ", x }'); do
      cp "$last" "$T/ns/$(printf %04d $i).png"
      i=$((i + 1))
    done
  done
}

walkthrough & trace & wait

ffmpeg -v error -y -framerate 24 -i "$T/wa/%04d.png" -filter_complex "[0]split[n][s];[n]crop=680:48:306:4[nav];[s]crop=680:624:506:76[st];[nav][st]vstack,tpad=stop_mode=clone:stop_duration=2,split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" "$P/docs/watermark_walkthrough.gif"
ffmpeg -v error -y -framerate 20 -i "$T/ns/%04d.png" -filter_complex "[0]scale=1000:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" "$P/docs/network_trace.gif"
rm -rf "$T"
ls -la "$P/docs"
