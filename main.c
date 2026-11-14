#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "canny.h"
#include "lane_assist.h"

/*
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
*/

int main(void) {
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
    const int width = 1280;
    const int height = 720;
    const size_t frame_bytes = (size_t)width * (size_t)height * 3;

    FILE *pipe_in = popen(
        "ffmpeg -hide_banner -loglevel error -i road.mp4 "
        "-vf \"scale=1280:720\" -f rawvideo -pix_fmt rgb24 -", "r"
    );

    FILE *pipe_out = popen(
        "ffmpeg -y -hide_banner -loglevel error "
        "-f rawvideo -pix_fmt rgb24 -s 1280x720 -r 30 -i - "
        "-c:v libx264 -pix_fmt yuv420p processed_road.mp4", "w"
    );

    if (!pipe_in || !pipe_out) {
        printf("Failed to open FFmpeg pipes!\n");
        return 1;
    }

    uint8_t *frame_rgb  = malloc(frame_bytes);
    uint8_t *img_gray   = malloc((size_t)width * (size_t)height);
    uint8_t *output_img = malloc((size_t)width * (size_t)height);

    int frame_count = 0;

    while (fread(frame_rgb, 1, frame_bytes, pipe_in) == frame_bytes) {
        rgbToGrayscale(frame_rgb, img_gray, width, height, 3);
        cannyEdge(img_gray, output_img, width, height);
        roi_mask(output_img, width, height);
        hough_transform(output_img, frame_rgb, width, height);

        fwrite(frame_rgb, 1, frame_bytes, pipe_out);
        frame_count++;
    }

    pclose(pipe_in);
    pclose(pipe_out);
    free(frame_rgb);
    free(img_gray);
    free(output_img);

    printf("Streamed and processed %d frames directly in memory!\n", frame_count);
    return 0;
}