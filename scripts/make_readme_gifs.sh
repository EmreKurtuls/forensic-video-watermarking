#!/bin/sh
set -e
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
P="$(cd "$(dirname "$0")/.." && pwd)"
T="$(mktemp -d)"
mkdir -p "$T/wa" "$T/ns" "$P/docs"
shot() { "$CHROME" --headless=new --disable-gpu --hide-scrollbars --window-size=1280,720 --virtual-time-budget="$3" --screenshot="$2" "$1" 2>/dev/null; }
for s in 1 2 3 4 5 6 7 8; do shot "file://$P/presentation/network_sim/index.html?step=$s" "$T/ns/$s.png" 6000; done
i=0
for f in $(awk 'BEGIN { for (x = 0; x < 120; x++) printf "%.2f ", x / 20 }'); do
  shot "file://$P/presentation/watermark_animation/index.html?freeze=$f" "$T/wa/$(printf %03d $i).png" 3000
  i=$((i + 1))
done
ffmpeg -v error -y -framerate 8 -i "$T/wa/%03d.png" -filter_complex "[0]split[n][s];[n]crop=680:48:306:4[nav];[s]crop=680:624:506:76[st];[nav][st]vstack,tpad=stop_mode=clone:stop_duration=2,split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" "$P/docs/watermark_walkthrough.gif"
ffmpeg -v error -y -framerate 1/3 -i "$T/ns/%d.png" -filter_complex "[0]fps=10,scale=1000:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" "$P/docs/network_trace.gif"
rm -rf "$T"
echo "wrote docs/watermark_walkthrough.gif and docs/network_trace.gif"
