# Real-Time Lane Detection Pipeline in Pure C (OpenMP + SIMD)

A high-performance, dependency-light computer vision pipeline written from scratch in C for real-time road lane detection. Built without OpenCV, the project implements a custom Canny edge detector, Region-of-Interest (ROI) masking, Hough Transform with inter-frame temporal smoothing, and zero-disk-I/O video streaming via FFmpeg pipes.

## Demo

https://github.com/user-attachments/assets/b4244360-0f6b-40fd-9643-c872dc90a973

*Detected lane boundaries with Exponential Moving Average (EMA) smoothing, dynamic vanishing-point cutoff, and lane-carpet rendering.*

---

## Architecture & Pipeline

1. **Zero-Disk Video Streaming (`main.c`):** Streams raw `rgb24` frames directly from FFmpeg `stdout` into RAM and pipes processed frames back into FFmpeg `stdin` for H.264 encoding.
2. **Color-Targeted Grayscale (`canny.c`):** Converts RGB to luminance using integer bit-shifts while boosting yellow lane markings to pure white (`255`) for higher contrast on faded asphalt.
3. **Gaussian Blur (`canny.c`):** Applies two 1D binomial passes (`[1, 4, 6, 4, 1] >> 4`) instead of one big `5x5 Matrix`
4. **Sobel Gradients & Fast Direction Quantization (`canny.c`):** Computes gradient magnitudes and assigns edge orientations using slope comparisons instead of `atan2f`.
5. **Non-Maximum Suppression & Iterative Hysteresis (`canny.c`):** Thins edges to 1-pixel width and traces connected edge chains using an explicit heap-allocated stack.
6. **Hough Transform & Temporal Smoothing (`lane_assist.c`):** Accumulates votes using pre-computed trigonometric lookup tables, blends frame-to-frame parameters (`80%` history / `20%` current frame) to eliminate flicker, and prevents line crossover above the vanishing point.

---

## Performance Benchmarks & Optimizations

Tested on a **41.1-second, 1280×720 @ 30 FPS video (1,232 frames)** on Apple Silicon (macOS, Clang `-O3 -march=native -ffast-math -fopenmp`).

### End-to-End Pipeline Speedup

| Pipeline Version | Total Wall-Clock Time | Effective Throughput | Real-Time Factor |
| :--- | :---: | :---: | :---: |
| **v1: Disk JPEG I/O (`stb_image` + `ffmpeg` extract/encode)** | `48.94 s` *(2.6s extract + 33.3s C + 13.0s encode)* | `25.2 FPS` | `0.84x` (Slower than real-time) |
| **v2: In-Memory FFmpeg `popen()` Raw RGB Pipes** | **`25.18 s`** *(full decode + vision + encode)* | **`48.9 FPS`** | **`1.63x` (Faster than real-time)** |

> **CPU Utilization:** In v2, total user CPU time reached `129.28s` compressed into `25.18s` wall-clock time—sustaining **~5.14 active CPU cores** concurrently across decoding, OpenMP vision kernels, and x264 encoding.

### Key Algorithmic & Systems Optimizations

| Optimization | Baseline Implementation | Optimized Implementation | Why It Is Faster |
| :--- | :--- | :--- | :--- |
| **Video I/O** | Extract `1,233` JPEGs to SSD, decode/encode per frame via `stb_image` | Direct `fread`/`fwrite` raw `rgb24` pipes via `popen()` | Eliminates ~70% of runtime spent on single-threaded software JPEG compression and SSD writes. |
| **Gaussian Blur** | 2D $5 \times 5$ convolution (`25` ops/pixel + float division) | Two 1D $5 \times 1$ passes (`10` ops/pixel + integer `>> 4`) | Reduces memory reads and multiplications by **2.5×** and replaces float division with a bit-shift. |
| **Edge Angle Quantization** | `atan2f(gy, gx) * (180 / M_PI)` per pixel | Slope threshold comparisons (`abs_gy <= 0.4142f * abs_gx`) | Replaces expensive multi-cycle transcendental trig approximations with single-cycle float multiplies. |
| **Memory Management** | `malloc`/`calloc` + `free` inside the per-frame loop | Persistent `static` buffers + `memset` reuse | Eliminates ~9,800 OS heap allocation/deallocation system calls across a 1,200-frame video. |
| **Hough Trigonometry** | `cosf`/`sinf` evaluated per edge pixel and angle | Pre-computed `180`-element lookup tables + OpenMP | Reduces millions of trig calls per frame to L1-cache array lookups. |

---

## Prerequisites & Usage

### Requirements
* **macOS / Linux / Windows** with `clang` (or `gcc`)
* **OpenMP** (`brew install libomp` on macOS)
* **FFmpeg** (`brew install ffmpeg` on macOS)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)NikitaSch2004/LaneDetector.git
   cd LaneDetector
   ./run_line_detector.sh

## Make sure the input video is named road.mp4 
