#!/bin/bash
# Render all frames in 4 parallel chunks, then concatenate.
set -e
cd "$(dirname "$0")"
N=$(python3 -c "import json,math;print(math.ceil(json.load(open('edit_map.json'))['duration']*30))")
mkdir -p parts
K=4; STEP=$(( (N + K - 1) / K ))
pids=()
for i in $(seq 0 $((K-1))); do
  a=$((i*STEP)); b=$(( (i+1)*STEP < N ? (i+1)*STEP : N ))
  python3 reel.py $a $b parts/p$i.mp4 > parts/log$i.txt 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
printf "file 'p%d.mp4'\n" 0 1 2 3 > parts/list.txt
ffmpeg -v error -y -f concat -safe 0 -i parts/list.txt -c copy video_noaudio.mp4
echo "frames=$N done"
