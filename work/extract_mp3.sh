#!/bin/bash
# Extract the audio track of a video (MP4/MOV/MKV/WEBM) to MP3 and discard the picture.
# The reel pipeline then uses ONLY this MP3 (work/source.mp3); no video frames are used.
#
#   usage: work/extract_mp3.sh <input-video-or-audio> [output.mp3]   (default: work/source.mp3)
#
# - picks the first audio stream, drops video/subtitle/data streams (-vn -sn -dn)
# - keeps the original sample rate, VBR ~245 kb/s (LAME -q:a 0) so speech quality is not degraded
# - if the input is already an MP3 it is copied as-is (no re-encode)
set -euo pipefail
in="${1:?usage: extract_mp3.sh <input> [output.mp3]}"
out="${2:-$(dirname "$0")/source.mp3}"

if ! ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "$in" | grep -q .; then
  echo "ERROR: no audio stream in $in" >&2
  exit 1
fi

codec=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "$in")
if [ "$codec" = "mp3" ] && [[ "${in,,}" == *.mp3 ]]; then
  cp "$in" "$out"
else
  ffmpeg -v error -y -i "$in" -map 0:a:0 -vn -sn -dn -c:a libmp3lame -q:a 0 "$out"
fi

din=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$in")
dout=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out")
echo "input : $in  (${din}s, audio codec: $codec)"
echo "output: $out (${dout}s)"
ffprobe -v error -show_entries stream=codec_type,codec_name,sample_rate,channels,bit_rate -of compact "$out"
