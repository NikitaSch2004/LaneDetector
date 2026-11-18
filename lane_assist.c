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

    /*SLOWER WAY, EVERY PIXEL IS COMPARED AGAINST THE TWO BOUNDS
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
    */

    //FASTER WAY, THE BOUNDS ARE TURNED INTO PIXEL INDICES ONCE PER ROW AND THE OUTSIDE PARTS ARE CLEARED WITH MEMSET
    #pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        uint8_t *row = input + (size_t)y * (size_t)width;

        if ((float)y < y_top) {
            memset(row, 0, (size_t)width);
            continue;
        }

        float t = ((float)y - y_top) / ((float)height - y_top);
        float left_bound  = (0.40f - 0.35f * t) * (float)width;
        float right_bound = (0.60f + 0.35f * t) * (float)width;

        int left_end    = (int)floorf(left_bound) + 1;
        int right_start = (int)ceilf(right_bound);
        if (left_end < 0) left_end = 0;
        if (left_end > width) left_end = width;
        if (right_start < 0) right_start = 0;
        if (right_start > width) right_start = width;

        memset(row, 0, (size_t)left_end);
        memset(row + right_start, 0, (size_t)(width - right_start));
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
    static float *edge_x = NULL;
    static float *edge_y = NULL;
    static float cos_table[180];
    static float sin_table[180];
    static int tables_initialized = 0;

    if (!accum) {
        accum  = calloc(accum_size, sizeof(*accum));
        edge_x = malloc((size_t)width * (size_t)height * sizeof(*edge_x));
        edge_y = malloc((size_t)width * (size_t)height * sizeof(*edge_y));
        if (!accum || !edge_x || !edge_y) {
            printf("Memory allocation failed in hough_transform!\n");
            free(accum);  accum = NULL;
            free(edge_x); edge_x = NULL;
            free(edge_y); edge_y = NULL;
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

    int y_start = (int)((float)height * 0.55f);

    /*SLOWER WAY, EVERY VOTE IS AN ATOMIC INCREMENT AND ALL THREADS FIGHT OVER THE SAME ACCUMULATOR CELLS
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
    */

    //FASTER WAY, EDGE PIXELS ARE COLLECTED ONCE AND EVERY THREAD OWNS WHOLE THETA ROWS OF THE ACCUMULATOR, SO NO ATOMICS ARE NEEDED
    int num_edges = 0;
    for (int y = y_start; y < height; y++)
    {
        const uint8_t *row = input + (size_t)y * (size_t)width;
        for (int x = 0; x < width; x++)
        {
            if (row[x] == 255)
            {
                edge_x[num_edges] = (float)x;
                edge_y[num_edges] = (float)y;
                num_edges++;
            }
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int theta = 5; theta <= 175; theta++)
    {
        if (theta > 75 && theta < 105) continue;

        int *restrict votes_row = accum + (size_t)theta * (size_t)num_rho + (size_t)max_dist;
        const float c = cos_table[theta];
        const float s = sin_table[theta];

        for (int i = 0; i < num_edges; i++)
        {
            int rho = (int)roundf(edge_x[i] * c + edge_y[i] * s);
            votes_row[rho]++;
        }
    }

    float best_left_theta = 0.0f, best_left_dist = 0.0f;
    float best_right_theta = 0.0f, best_right_dist = 0.0f;
    int max_left_votes = 0, max_right_votes = 0;
    int best_left_rho = 0, best_right_rho = 0;

    for (int theta = 5; theta <= 175; theta++)
    {
        if (theta > 75 && theta < 105) continue;

        const int *votes_row = accum + (size_t)theta * (size_t)num_rho;
        int is_left = theta <= 75;
        int *max_votes = is_left ? &max_left_votes : &max_right_votes;
        int *best_rho  = is_left ? &best_left_rho  : &best_right_rho;

        for (int rho_idx = 0; rho_idx < num_rho; rho_idx++)
        {
            int votes = votes_row[rho_idx];
            if (votes > *max_votes || (votes == *max_votes && votes > 0 && rho_idx < *best_rho))
            {
                *max_votes = votes;
                *best_rho  = rho_idx;
                if (is_left) best_left_theta  = (float)theta;
                else         best_right_theta = (float)theta;
            }
        }
    }
    best_left_dist  = (float)(best_left_rho  - max_dist);
    best_right_dist = (float)(best_right_rho - max_dist);

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
