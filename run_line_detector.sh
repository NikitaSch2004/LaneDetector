#!/bin/bash

set -e

cd "$(dirname "$0")"

INPUT_VIDEO="${1:-road.mp4}"
OUTPUT_VIDEO="${2:-processed_road.mp4}"

./build.sh

echo "=========================================="
echo " Running Lane Detection Pipeline"
echo "=========================================="
time ./lane_detector "$INPUT_VIDEO" "$OUTPUT_VIDEO"

echo "=========================================="
echo " Done! Opening $OUTPUT_VIDEO..."
echo "=========================================="
if [[ "$(uname)" == "Darwin" ]]; then
  open "$OUTPUT_VIDEO"
elif command -v xdg-open > /dev/null; then
  xdg-open "$OUTPUT_VIDEO"
fi
