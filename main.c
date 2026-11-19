#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>
#include <omp.h>
#include "canny.h"
#include "lane_assist.h"

/*
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
*/

/*SLOWER WAY OF ENCODING, LIBX264 ON THE DEFAULT PRESET WAS THE BOTTLENECK OF THE WHOLE PIPELINE
#define VIDEO_ENCODER "-c:v libx264"
*/

//FASTER WAY, HARDWARE ENCODER ON MACOS AND A FASTER X264 PRESET EVERYWHERE ELSE
#ifdef __APPLE__
#define VIDEO_ENCODER "-c:v h264_videotoolbox -b:v 8M"
#else
#define VIDEO_ENCODER "-c:v libx264 -preset veryfast"
#endif

#define PROCESSING_WIDTH 1280
#define RING_SIZE 4
#define CANNY_ROW_MARGIN 32

enum { SLOT_FREE, SLOT_READ, SLOT_PROCESSED, SLOT_END };

typedef struct {
    FILE *pipe_in;
    FILE *pipe_out;
    size_t frame_bytes;
    uint8_t *frames[RING_SIZE];
    int state[RING_SIZE];
    int stop;
    int write_failed;
    pthread_mutex_t lock;
    pthread_cond_t changed;
} FrameRing;

static int ring_wait(FrameRing *ring, int slot, int wanted, int alternative)
{
    pthread_mutex_lock(&ring->lock);
    while (ring->state[slot] != wanted && ring->state[slot] != alternative && !ring->stop) {
        pthread_cond_wait(&ring->changed, &ring->lock);
    }
    int state = ring->stop ? SLOT_END : ring->state[slot];
    pthread_mutex_unlock(&ring->lock);
    return state;
}

static void ring_set(FrameRing *ring, int slot, int state)
{
    pthread_mutex_lock(&ring->lock);
    ring->state[slot] = state;
    pthread_cond_broadcast(&ring->changed);
    pthread_mutex_unlock(&ring->lock);
}

static void ring_stop(FrameRing *ring)
{
    pthread_mutex_lock(&ring->lock);
    ring->stop = 1;
    pthread_cond_broadcast(&ring->changed);
    pthread_mutex_unlock(&ring->lock);
}

static void *reader_thread(void *arg)
{
    FrameRing *ring = arg;
    for (long i = 0; ; i++) {
        int slot = (int)(i % RING_SIZE);
        if (ring_wait(ring, slot, SLOT_FREE, SLOT_FREE) != SLOT_FREE) return NULL;

        if (fread(ring->frames[slot], 1, ring->frame_bytes, ring->pipe_in) != ring->frame_bytes) {
            ring_set(ring, slot, SLOT_END);
            return NULL;
        }
        ring_set(ring, slot, SLOT_READ);
    }
}

static void *writer_thread(void *arg)
{
    FrameRing *ring = arg;
    for (long i = 0; ; i++) {
        int slot = (int)(i % RING_SIZE);
        if (ring_wait(ring, slot, SLOT_PROCESSED, SLOT_END) != SLOT_PROCESSED) return NULL;

        if (fwrite(ring->frames[slot], 1, ring->frame_bytes, ring->pipe_out) != ring->frame_bytes) {
            ring->write_failed = 1;
            ring_stop(ring);
            return NULL;
        }
        ring_set(ring, slot, SLOT_FREE);
    }
}

int main(int argc, char **argv) {
    /*READS ONE JPG AT A TIME PROCCESSES AND WRITES IT AGAIN ON DISK, VERY SLOW
    char in_path[256];
    char out_path[256];
    int frame_idx = 1;

    uint8_t *img_gray   = NULL;
    uint8_t *output_img = NULL;
    while (1) {
        snprintf(in_path, sizeof(in_path), "frames_in/frame_%04d.jpg", frame_idx);

        int width, height, channels;
        unsigned char *img = stbi_load(in_path, &width, &height, &channels, 3);
        if (img == NULL) {
            break;
        }
        if (!img_gray && !output_img) {
            size_t total_pixels = (size_t)width * (size_t)height;
            img_gray   = malloc(total_pixels);
            output_img = malloc(total_pixels);
        }

        if (img_gray && output_img) {
            rgbToGrayscale(img, img_gray, width, height, 3);
            cannyEdge(img_gray, output_img, width, height);
            roi_mask(output_img, width, height);
            hough_transform(output_img, img, width, height);

            snprintf(out_path, sizeof(out_path), "frames_out/out_%04d.jpg", frame_idx);
            stbi_write_jpg(out_path, width, height, 3, img, 90);
            if (frame_idx % 30 == 0) printf("Processed frame %d...\r", frame_idx);
        }

        stbi_image_free(img);
        frame_idx++;
    }

    printf("Finished processing %d frames!\n", frame_idx - 1);
    free(img_gray);
    free(output_img);
    return 0;
    */

    //MUCH FASTER, CONNECT TO FFMPEG PIPELINE AND STORES IN RAM FRAMES INSTEAD OF REDING FROM A JPG ON DISK
    const char *input_path  = argc > 1 ? argv[1] : "road.mp4";
    const char *output_path = argc > 2 ? argv[2] : "processed_road.mp4";

    if (strchr(input_path, '\'') || strchr(output_path, '\'')) {
        fprintf(stderr, "File names must not contain single quotes!\n");
        return 1;
    }

    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "ffprobe -v error -select_streams v:0 "
        "-show_entries stream=width,height,r_frame_rate -of csv=p=0 '%s'", input_path);

    FILE *probe = popen(cmd, "r");
    char probe_line[128] = {0};
    int src_width = 0, src_height = 0;
    char frame_rate[32] = {0};
    if (probe) {
        if (fgets(probe_line, sizeof(probe_line), probe)) {
            sscanf(probe_line, "%d,%d,%31[0-9/]", &src_width, &src_height, frame_rate);
        }
        pclose(probe);
    }

    if (src_width <= 0 || src_height <= 0 || frame_rate[0] == '\0') {
        fprintf(stderr, "Could not read video info from %s (is FFmpeg installed and does the file exist?)\n", input_path);
        return 1;
    }

    const int width = PROCESSING_WIDTH;
    const int height = (int)((double)src_height * width / src_width / 2.0 + 0.5) * 2;
    const size_t frame_bytes = (size_t)width * (size_t)height * 3;

#ifdef SIGPIPE
    signal(SIGPIPE, SIG_IGN);
#endif

    snprintf(cmd, sizeof(cmd),
        "ffmpeg -hide_banner -loglevel error -i '%s' "
        "-vf \"scale=%d:%d\" -f rawvideo -pix_fmt rgb24 -", input_path, width, height);
    FILE *pipe_in = popen(cmd, "r");

    snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -hide_banner -loglevel error "
        "-f rawvideo -pix_fmt rgb24 -s %dx%d -r %s -i - "
        VIDEO_ENCODER " -pix_fmt yuv420p '%s'", width, height, frame_rate, output_path);
    FILE *pipe_out = popen(cmd, "w");

    if (!pipe_in || !pipe_out) {
        fprintf(stderr, "Failed to open FFmpeg pipes!\n");
        if (pipe_in)  pclose(pipe_in);
        if (pipe_out) pclose(pipe_out);
        return 1;
    }

    /*SLOWER WAY, READING, PROCESSING AND WRITING HAPPEN ONE AFTER ANOTHER, SO THE VISION CODE WAITS FOR FFMPEG AND FFMPEG WAITS FOR IT
    uint8_t *frame_rgb  = malloc(frame_bytes);
    uint8_t *img_gray   = malloc((size_t)width * (size_t)height);
    uint8_t *output_img = malloc((size_t)width * (size_t)height);

    if (!frame_rgb || !img_gray || !output_img) {
        fprintf(stderr, "Memory allocation failed in main!\n");
        free(frame_rgb);
        free(img_gray);
        free(output_img);
        pclose(pipe_in);
        pclose(pipe_out);
        return 1;
    }

    int frame_count = 0;
    int write_failed = 0;
    double vision_seconds = 0.0;

    while (fread(frame_rgb, 1, frame_bytes, pipe_in) == frame_bytes) {
        double start = omp_get_wtime();
        rgbToGrayscale(frame_rgb, img_gray, width, height, 3);
        cannyEdge(img_gray, output_img, width, height);
        roi_mask(output_img, width, height);
        hough_transform(output_img, frame_rgb, width, height);
        vision_seconds += omp_get_wtime() - start;

        if (fwrite(frame_rgb, 1, frame_bytes, pipe_out) != frame_bytes) {
            write_failed = 1;
            break;
        }
        frame_count++;
    }

    pclose(pipe_in);
    int encoder_status = pclose(pipe_out);
    free(frame_rgb);
    free(img_gray);
    free(output_img);
    */

    //FASTER WAY, A READER THREAD AND A WRITER THREAD SHARE A RING OF FRAMES WITH THE VISION CODE, SO DECODING, PROCESSING AND ENCODING OVERLAP
    const int first_row = roi_first_row(height) - CANNY_ROW_MARGIN;
    int failed = 1;
    int frame_count = 0;
    double vision_seconds = 0.0;

    FrameRing ring = { .pipe_in = pipe_in, .pipe_out = pipe_out, .frame_bytes = frame_bytes };
    pthread_mutex_init(&ring.lock, NULL);
    pthread_cond_init(&ring.changed, NULL);

    uint8_t *img_gray   = malloc((size_t)width * (size_t)height);
    uint8_t *output_img = malloc((size_t)width * (size_t)height);
    int alloc_failed = !img_gray || !output_img;
    for (int i = 0; i < RING_SIZE; i++) {
        ring.frames[i] = malloc(frame_bytes);
        if (!ring.frames[i]) alloc_failed = 1;
    }

    pthread_t reader, writer;
    if (alloc_failed) {
        fprintf(stderr, "Memory allocation failed in main!\n");
        goto cleanup;
    }
    if (pthread_create(&reader, NULL, reader_thread, &ring) != 0) {
        fprintf(stderr, "Failed to start the reader thread!\n");
        goto cleanup;
    }
    if (pthread_create(&writer, NULL, writer_thread, &ring) != 0) {
        fprintf(stderr, "Failed to start the writer thread!\n");
        ring_stop(&ring);
        pthread_join(reader, NULL);
        goto cleanup;
    }

    for (long i = 0; ; i++) {
        int slot = (int)(i % RING_SIZE);
        if (ring_wait(&ring, slot, SLOT_READ, SLOT_END) != SLOT_READ) break;

        uint8_t *frame_rgb = ring.frames[slot];
        double start = omp_get_wtime();
        rgbToGrayscale(frame_rgb, img_gray, width, height, 3);
        cannyEdge(img_gray, output_img, width, height, first_row);
        roi_mask(output_img, width, height);
        hough_transform(output_img, frame_rgb, width, height);
        vision_seconds += omp_get_wtime() - start;

        ring_set(&ring, slot, SLOT_PROCESSED);
        frame_count++;
    }

    pthread_join(reader, NULL);
    pthread_join(writer, NULL);
    failed = 0;

cleanup:
    pclose(pipe_in);
    int encoder_status = pclose(pipe_out);
    free(img_gray);
    free(output_img);
    for (int i = 0; i < RING_SIZE; i++) free(ring.frames[i]);
    pthread_mutex_destroy(&ring.lock);
    pthread_cond_destroy(&ring.changed);

    if (failed) return 1;
    if (frame_count == 0) {
        fprintf(stderr, "No frames were read from %s!\n", input_path);
        return 1;
    }
    if (ring.write_failed || encoder_status != 0) {
        fprintf(stderr, "FFmpeg failed while encoding %s!\n", output_path);
        return 1;
    }

    double ms_per_frame = 1000.0 * vision_seconds / frame_count;
    printf("Streamed and processed %d frames directly in memory!\n", frame_count);
    printf("Vision pipeline: %.2f ms/frame (%.0f FPS)\n", ms_per_frame, 1000.0 / ms_per_frame);
    return 0;
}
