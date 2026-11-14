#!/bin/bash

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo " Compiling C pipeline (OpenMP + SIMD)"
echo "=========================================="
LIBOMP_PREFIX=$(brew --prefix libomp)

clang -O3 -march=native -ffast-math -Xpreprocessor -fopenmp \
  -I"$LIBOMP_PREFIX/include" \
  -L"$LIBOMP_PREFIX/lib" -lomp \
  main.c canny.c lane_assist.c \
  -o lane_detector

echo "=========================================="
echo " Running Lane Detection Pipeline"
echo "=========================================="
time ./lane_detector

echo "=========================================="
echo " Done! Opening $OUTPUT_VIDEO..."
echo "=========================================="
open "$OUTPUT_VIDEO"