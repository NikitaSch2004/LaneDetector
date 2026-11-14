#include "lane_assist.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void roi_mask(uint8_t *input, int width, int height)
{
    if (!input) return;

    float y_top = (float)height * 0.55f;

    #pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        if ((float)y < y_top) {
            for (int x = 0; x < width; x++) {
                input[y * width + x] = 0;
            }
        } else {
            float t = ((float)y - y_top) / ((float)height - y_top);
            float left_bound  = (0.40f - 0.35f * t) * (float)width;
            float right_bound = (0.60f + 0.35f * t) * (float)width;

            for (int x = 0; x < width; x++) {
                if ((float)x <= left_bound || (float)x >= right_bound) {
                    input[y * width + x] = 0;
                }
            }
        }
    }
}

void hough_transform(const uint8_t *input, unsigned char *draw, int width, int height)
{
    static float smooth_left_theta = 0.0f, smooth_left_dist = 0.0f;
    static float smooth_right_theta = 0.0f, smooth_right_dist = 0.0f;
    static int has_left = 0, has_right = 0;

    if (!input || !draw) return;

    const int max_dist = (int)ceilf(sqrtf((float)width * (float)width + (float)height * (float)height));
    const int num_rho = 2 * max_dist + 1;
    size_t accum_size = (size_t)num_rho * 180;
    static int *accum = NULL;
    static float cos_table[180];
    static float sin_table[180];
    static int tables_initialized = 0;

    if (!accum) {
        accum = malloc(accum_size * sizeof(*accum));
        if (!accum) {
            printf("Memory allocation failed in hough_transform!\n");
            return;
        }
    }

    if (!tables_initialized) {
        for (int theta = 0; theta < 180; theta++) {
            float rad = (float)theta * ((float)M_PI / 180.0f);
            cos_table[theta] = cosf(rad);
            sin_table[theta] = sinf(rad);
        }
        tables_initialized = 1;
    }

    for (int theta = 0; theta < 180; theta++) {
        float rad = (float)theta * ((float)M_PI / 180.0f);
        cos_table[theta] = cosf(rad);
        sin_table[theta] = sinf(rad);
    }

    int y_start = (int)((float)height * 0.55f);

    #pragma omp parallel for collapse(2)
    for (int y = y_start; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (input[y * width + x] == 255)
            {
                for (int theta = 5; theta <= 175; theta++)
                {
                    if (theta > 75 && theta < 105) continue;

                    int rho = (int)roundf((float)x * cos_table[theta] + (float)y * sin_table[theta]);
                    #pragma omp atomic
                    accum[(rho + max_dist) * 180 + theta]++;
                }
            }
        }
    }

    float best_left_theta = 0.0f, best_left_dist = 0.0f;
    float best_right_theta = 0.0f, best_right_dist = 0.0f;
    int max_left_votes = 0, max_right_votes = 0;

    for (int rho_idx = 0; rho_idx < num_rho; rho_idx++)
    {
        for (int theta = 5; theta <= 175; theta++)
        {
            int votes = accum[rho_idx * 180 + theta];
            if (theta >= 5 && theta <= 75 && votes > max_left_votes)
            {
                best_left_theta = (float)theta;
                best_left_dist  = (float)(rho_idx - max_dist);
                max_left_votes  = votes;
            }
            else if (theta >= 105 && votes > max_right_votes)
            {
                best_right_theta = (float)theta;
                best_right_dist  = (float)(rho_idx - max_dist);
                max_right_votes  = votes;
            }
        }
    }

    if (max_left_votes > 20) {
        if (!has_left) {
            smooth_left_theta = best_left_theta;
            smooth_left_dist  = best_left_dist;
            has_left = 1;
        } else {
            smooth_left_theta = 0.8f * smooth_left_theta + 0.2f * best_left_theta;
            smooth_left_dist  = 0.8f * smooth_left_dist  + 0.2f * best_left_dist;
        }
    }

    if (max_right_votes > 20) {
        if (!has_right) {
            smooth_right_theta = best_right_theta;
            smooth_right_dist  = best_right_dist;
            has_right = 1;
        } else {
            smooth_right_theta = 0.8f * smooth_right_theta + 0.2f * best_right_theta;
            smooth_right_dist  = 0.8f * smooth_right_dist  + 0.2f * best_right_dist;
        }
    }

    float l_rad = smooth_left_theta * ((float)M_PI / 180.0f);
    float r_rad = smooth_right_theta * ((float)M_PI / 180.0f);
    float l_sin = sinf(l_rad), l_cos = cosf(l_rad);
    float r_sin = sinf(r_rad), r_cos = cosf(r_rad);

    for (int y = y_start; y < height; y++)
    {
        int x_left  = -1;
        int x_right = -1;

        if (has_left && fabsf(l_cos) > 0.001f) {
            x_left = (int)roundf((smooth_left_dist - (float)y * l_sin) / l_cos);
        }
        if (has_right && fabsf(r_cos) > 0.001f) {
            x_right = (int)roundf((smooth_right_dist - (float)y * r_sin) / r_cos);
        }

        if (has_left && has_right && x_left >= (x_right - 10)) {
            continue;
        }

        if (has_left && has_right) {
            int fill_start = (x_left + 3 > 0) ? (x_left + 3) : 0;
            int fill_end   = (x_right - 2 < width) ? (x_right - 2) : width;

            for (int x = fill_start; x < fill_end; x++) {
                int rgb_idx = (y * width + x) * 3;
                draw[rgb_idx]     = (uint8_t)((draw[rgb_idx]     * 3) >> 2);
                draw[rgb_idx + 1] = (uint8_t)((draw[rgb_idx + 1] * 3 + 255) >> 2);
                draw[rgb_idx + 2] = (uint8_t)((draw[rgb_idx + 2] * 3) >> 2);
            }
        }

        for (int dx = -2; dx <= 2; dx++) {
            if (has_left) {
                int nx = x_left + dx;
                if (nx >= 0 && nx < width) {
                    int rgb_idx = (y * width + nx) * 3;
                    draw[rgb_idx]     = 0;
                    draw[rgb_idx + 1] = 255;
                    draw[rgb_idx + 2] = 0;
                }
            }
            if (has_right) {
                int nx = x_right + dx;
                if (nx >= 0 && nx < width) {
                    int rgb_idx = (y * width + nx) * 3;
                    draw[rgb_idx]     = 0;
                    draw[rgb_idx + 1] = 255;
                    draw[rgb_idx + 2] = 0;
                }
            }
        }
    }

    memset(accum, 0, accum_size * sizeof(*accum));
}