#!/bin/bash
# Two-pass EBU R128 loudness normalisation (-14 LUFS / -1 dBTP) and mux with the rendered picture.
set -e
cd "$(dirname "$0")"
J=$(ffmpeg -hide_banner -i mix_pre.wav -af loudnorm=I=-14:TP=-1.0:LRA=9:print_format=json -f null - 2>&1 | sed -n '/{/,/}/p')
gi(){ echo "$J" | python3 -c "import json,sys;print(json.load(sys.stdin)['$1'])"; }
ffmpeg -v error -y -i mix_pre.wav -af "loudnorm=I=-14:TP=-1.0:LRA=9:measured_I=$(gi input_i):measured_TP=$(gi input_tp):measured_LRA=$(gi input_lra):measured_thresh=$(gi input_thresh):offset=$(gi target_offset):linear=true,aresample=48000" -c:a pcm_s24le mix_master.wav
mkdir -p ../output
V=${VIDEO:-video_noaudio.mp4}
ffmpeg -v error -y -i "$V" -i mix_master.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 256k -ar 48000 -shortest -movflags +faststart ../output/space_atau_avoidance_reel_1080x1920.mp4
ffmpeg -hide_banner -i ../output/space_atau_avoidance_reel_1080x1920.mp4 -af ebur128=peak=true -f null - 2>&1 | grep -E "^\s+(I|LRA|Peak):"
