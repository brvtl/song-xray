#!/usr/bin/env bash
# Download a public-domain recording for a real-music test (the synthetic clips are not realistic):
#   "The Stars and Stripes Forever" (J. P. Sousa), United States Marine Band, 2017 — a work of the U.S. federal government,
#   public domain. Source: Wikimedia Commons / marineband.marines.mil ("The Complete Marches of John Philip Sousa").
# Writes clips/sousa_30s.wav (44.1 kHz) and clips/sousa_60s.mp3 (excerpts from 0:05).
set -euo pipefail
cd "$(dirname "$0")/.."
URL='https://upload.wikimedia.org/wikipedia/commons/e/ed/Sousa%27s_%22The_Stars_and_Stripes_Forever%22_-_United_States_Marine_Band_%282017%29.mp3'
mkdir -p clips/download
[ -f clips/download/sousa_full.mp3 ] || curl -sSfL -A "song-xray-demo/0.1 (local demo)" -o clips/download/sousa_full.mp3 "$URL"
ffmpeg -v error -y -ss 5 -t 30 -i clips/download/sousa_full.mp3 -ac 1 -ar 44100 clips/sousa_30s.wav
ffmpeg -v error -y -ss 5 -t 60 -i clips/download/sousa_full.mp3 -c:a libmp3lame -b:a 192k clips/sousa_60s.mp3
ls -la clips/sousa_30s.wav clips/sousa_60s.mp3
