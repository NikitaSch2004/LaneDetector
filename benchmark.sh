#!/bin/bash

set -e

cd "$(dirname "$0")"

RUNS="${RUNS:-3}"
OUTPUT_DIR=benchmark_output

./build.sh
mkdir -p "$OUTPUT_DIR"

VIDEOS=()
[[ -f road.mp4 ]] && VIDEOS+=(road.mp4)
for video in test_videos/*/*.mp4; do
  [[ "$(basename "$video")" == processed_* ]] && continue
  VIDEOS+=("$video")
done

if [[ ${#VIDEOS[@]} -eq 0 ]]; then
  echo "No videos found (road.mp4 or test_videos/*/*.mp4)."
  exit 1
fi

median() {
  sort -n | awk '{ v[NR] = $1 } END { print v[int((NR + 1) / 2)] }'
}

LOG=$(mktemp)
TIMES=$(mktemp)
trap 'rm -f "$LOG" "$TIMES" "$TIMES.vision" "$TIMES.total"' EXIT

echo "=========================================="
echo " Benchmark: ${#VIDEOS[@]} videos, median of $RUNS runs"
echo "=========================================="
printf "%-26s %-10s %7s %9s %11s %10s %9s %11s\n" \
  "Video" "Input" "Frames" "Length" "Vision" "Vision" "Total" "Realtime"
printf "%-26s %-10s %7s %9s %11s %10s %9s %11s\n" \
  "" "" "" "(s)" "(ms/frame)" "(FPS)" "(s)" "factor"

for video in "${VIDEOS[@]}"; do
  name=$(basename "$video" .mp4)
  output="$OUTPUT_DIR/processed_$name.mp4"
  input_size=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$video")
  length=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$video")

  : > "$TIMES.vision"
  : > "$TIMES.total"
  for ((run = 1; run <= RUNS; run++)); do
    TIMEFORMAT=%R
    if ! { time ./lane_detector "$video" "$output" > "$LOG" 2>&1; } 2> "$TIMES"; then
      echo "lane_detector failed on $video:"
      cat "$LOG"
      exit 1
    fi
    awk '/Vision pipeline/ { print $3 }' "$LOG" >> "$TIMES.vision"
    tail -n 1 "$TIMES" >> "$TIMES.total"
  done

  frames=$(awk '/Streamed and processed/ { print $4 }' "$LOG")
  vision_ms=$(median < "$TIMES.vision")
  total_s=$(median < "$TIMES.total")

  awk -v n="$name" -v s="$input_size" -v f="$frames" -v len="$length" -v v="$vision_ms" -v t="$total_s" 'BEGIN {
    printf "%-26s %-10s %7d %9.1f %11.2f %10.0f %9.2f %10.1fx\n", n, s, f, len, v, 1000 / v, t, len / t
  }'
done
