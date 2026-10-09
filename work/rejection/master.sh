#!/bin/bash
# Two-pass loudness normalisation (-14 LUFS / -1 dBTP), 2-pass H.264 ~5 Mbps + AAC 256k, 720p preview.
set -e
cd "$(dirname "$0")"
SLUG=rejection_cemburu
OUT=../../output/$SLUG
mkdir -p "$OUT"
J=$(ffmpeg -hide_banner -i mix_pre.wav -af loudnorm=I=-14:TP=-1.0:LRA=9:print_format=json -f null - 2>&1 | sed -n '/{/,/}/p')
gi(){ echo "$J" | python3 -c "import json,sys;print(json.load(sys.stdin)['$1'])"; }
ffmpeg -v error -y -i mix_pre.wav -af "loudnorm=I=-14:TP=-1.0:LRA=9:measured_I=$(gi input_i):measured_TP=$(gi input_tp):measured_LRA=$(gi input_lra):measured_thresh=$(gi input_thresh):offset=$(gi target_offset):linear=true,aresample=48000" -c:a pcm_s24le mix_master.wav
V=${VIDEO:-video_noaudio.mp4}
ffmpeg -v error -y -i "$V" -c:v libx264 -preset slow -b:v 5000k -maxrate 7000k -bufsize 10000k -r 30 -pix_fmt yuv420p \
  -pass 1 -passlogfile parts/x264 -an -f mp4 /dev/null
ffmpeg -v error -y -i "$V" -i mix_master.wav -map 0:v -map 1:a -c:v libx264 -preset slow -b:v 5000k -maxrate 7000k \
  -bufsize 10000k -r 30 -pix_fmt yuv420p -pass 2 -passlogfile parts/x264 -c:a aac -b:a 256k -ar 48000 -shortest \
  -movflags +faststart "$OUT/${SLUG}_1080x1920.mp4"
ffmpeg -v error -y -i "$OUT/${SLUG}_1080x1920.mp4" -vf scale=720:1280:flags=lanczos -c:v libx264 -preset slow \
  -b:v 1150k -maxrate 1600k -bufsize 2300k -c:a aac -b:a 128k -movflags +faststart "$OUT/preview_720p.mp4"
ls -la "$OUT"
ffmpeg -hide_banner -i "$OUT/${SLUG}_1080x1920.mp4" -af ebur128=peak=true -f null - 2>&1 | grep -E "^\s+(I|LRA|Peak):"
