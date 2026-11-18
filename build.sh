#!/bin/bash

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo " Compiling C pipeline (OpenMP + SIMD)"
echo "=========================================="
if [[ "$(uname)" == "Darwin" ]]; then
  LIBOMP_PREFIX=$(brew --prefix libomp)

  clang -O3 -march=native -ffast-math -Wall -Wextra -Xpreprocessor -fopenmp \
    -I"$LIBOMP_PREFIX/include" \
    -L"$LIBOMP_PREFIX/lib" -lomp \
    main.c canny.c lane_assist.c \
    -o lane_detector
else
  "${CC:-gcc}" -O3 -march=native -ffast-math -Wall -Wextra -fopenmp \
    main.c canny.c lane_assist.c \
    -lm -o lane_detector
fi
