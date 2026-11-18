#include "canny.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void rgbToGrayscale(const uint8_t *img, uint8_t *output, const int width, const int height, const int channels)
{
    if (!img || !output) return;

    /*SLOWER WAY, COLLAPSE(2) WITHOUT RESTRICT POINTERS STOPS THE COMPILER FROM VECTORIZING THE LOOP
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t gray_idx = (size_t)y * (size_t)width + (size_t)x;
            size_t pixel_index = gray_idx * (size_t)channels;

            unsigned char r = img[pixel_index];
            unsigned char g = img[pixel_index + 1];
            unsigned char b = img[pixel_index + 2];

            uint8_t gray = (uint8_t)((306 * r + 601 * g + 117 * b) >> 10);

            if (r > 140 && g > 120 && b < 100 && (r - b) > 50) {
                gray = 255;
            }

            output[gray_idx] = gray;
        }
    }
    */

    //FASTER WAY, ONE ROW PER THREAD WITH RESTRICT POINTERS SO THE INNER LOOP IS VECTORIZED (SIMD)
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        const uint8_t *restrict src = img + (size_t)y * (size_t)width * (size_t)channels;
        uint8_t *restrict dst = output + (size_t)y * (size_t)width;

        #pragma omp simd
        for (int x = 0; x < width; x++) {
            int r = src[x * channels];
            int g = src[x * channels + 1];
            int b = src[x * channels + 2];

            int gray = (306 * r + 601 * g + 117 * b) >> 10;
            int yellow = r > 140 && g > 120 && b < 100 && (r - b) > 50;

            dst[x] = (uint8_t)(yellow ? 255 : gray);
        }
    }
}

void cannyEdge(const uint8_t *img, uint8_t *output, int width, int height)
{
    if (!img || !output) return;

    size_t total_pixels = (size_t)width * (size_t)height;
    static uint8_t *gaussian_blur = NULL;
    static float   *magnitude_arr = NULL;
    static uint8_t *angles        = NULL;
    static float   *nms           = NULL;

    if (!gaussian_blur) {
        gaussian_blur = calloc(total_pixels, sizeof(*gaussian_blur));
        magnitude_arr = calloc(total_pixels, sizeof(*magnitude_arr));
        angles        = calloc(total_pixels, sizeof(*angles));
        nms           = calloc(total_pixels, sizeof(*nms));

        if (!gaussian_blur || !magnitude_arr || !angles || !nms) {
            printf("Memory allocation failed in cannyEdge!\n");
            free(gaussian_blur); gaussian_blur = NULL;
            free(magnitude_arr); magnitude_arr = NULL;
            free(angles);        angles = NULL;
            free(nms);           nms = NULL;
            return;
        }
    }

    /*SLOWER WAY, CLEARING EVERY BUFFER ON EVERY FRAME EVEN THOUGH EACH STAGE OVERWRITES IT
    memset(gaussian_blur, 0, total_pixels * sizeof(*gaussian_blur));
    memset(magnitude_arr, 0, total_pixels * sizeof(*magnitude_arr));
    memset(angles,        0, total_pixels * sizeof(*angles));
    memset(nms,           0, total_pixels * sizeof(*nms));
    */

    gaussian_blur_conv(img, gaussian_blur, width, height);
    float max_mag = magnitude_and_angle(gaussian_blur, magnitude_arr, angles, width, height);
    non_maximum_suppression(magnitude_arr, angles, nms, width, height, max_mag);
    histeresis(nms, output, width, height);
}

void gaussian_blur_conv(const uint8_t *img, uint8_t *output, const int width, const int height)
{
    /*SLOWER WAY OF APPLYING GAUSSIAN KERNEL USING 5X5 MATRIX
    if (!img || !output) return;

    static const int gaussian_kernel[25] = {
        2,  4,  5,  4, 2,
        4,  9, 12,  9, 4,
        5, 12, 15, 12, 5,
        4,  9, 12,  9, 4,
        2,  4,  5,  4, 2
    };

    #pragma omp parallel for collapse(2)
    for (int y = 2; y < height - 2; y++)
    {
        for (int x = 2; x < width - 2; x++)
        {
            float sum = 0.0f;
            for (int ky = -2; ky <= 2; ky++)
            {
                for (int kx = -2; kx <= 2; kx++)
                {
                    int pixel_index = (y + ky) * width + (x + kx);
                    sum += (float)((int)img[pixel_index] * gaussian_kernel[(ky + 2) * 5 + (kx + 2)]);
                }
            }
            output[y * width + x] = (uint8_t)(sum / 159.0f);
        }
    }
    */

    /*SLOWER WAY OF APPLYING THE SEPARATED 5X1 KERNELS, COLLAPSE(2) STOPS VECTORIZATION AND THE ZERO BORDER CREATES FAKE EDGES
    if (!img || !output) return;
    size_t total_pixels = (size_t)width * (size_t)height;
    static uint8_t *temp = NULL;
    if (!temp) {
        temp = calloc(total_pixels, sizeof(*temp));
        if (!temp) return;
    }

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 2; x < width - 2; x++)
        {
            int row = y * width;
            int sum = img[row + (x - 2)] * 1
                    + img[row + (x - 1)] * 4
                    + img[row +  x     ] * 6
                    + img[row + (x + 1)] * 4
                    + img[row + (x + 2)] * 1;
            temp[row + x] = (uint8_t)(sum >> 4);
        }
    }

    #pragma omp parallel for collapse(2)
    for (int y = 2; y < height - 2; y++)
    {
        for (int x = 2; x < width - 2; x++)
        {
            int sum = temp[(y - 2) * width + x] * 1
                    + temp[(y - 1) * width + x] * 4
                    + temp[ y      * width + x] * 6
                    + temp[(y + 1) * width + x] * 4
                    + temp[(y + 2) * width + x] * 1;
            output[y * width + x] = (uint8_t)(sum >> 4);
        }
    }
    */

    //FASTER WAY, ONE ROW PER THREAD WITH RESTRICT POINTERS SO BOTH PASSES ARE VECTORIZED (SIMD), BORDER PIXELS ARE REPLICATED
    if (!img || !output || width < 4 || height < 4) return;
    size_t total_pixels = (size_t)width * (size_t)height;
    static uint8_t *temp = NULL;
    if (!temp) {
        temp = malloc(total_pixels * sizeof(*temp));
        if (!temp) return;
    }

    #pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        const uint8_t *restrict src = img + (size_t)y * (size_t)width;
        uint8_t *restrict dst = temp + (size_t)y * (size_t)width;

        #pragma omp simd
        for (int x = 2; x < width - 2; x++)
        {
            int sum = src[x - 2] * 1
                    + src[x - 1] * 4
                    + src[x    ] * 6
                    + src[x + 1] * 4
                    + src[x + 2] * 1;
            dst[x] = (uint8_t)(sum >> 4);
        }

        const int border_x[4] = { 0, 1, width - 2, width - 1 };
        for (int i = 0; i < 4; i++)
        {
            int x = border_x[i];
            int xm2 = x - 2 < 0 ? 0 : x - 2;
            int xm1 = x - 1 < 0 ? 0 : x - 1;
            int xp1 = x + 1 >= width ? width - 1 : x + 1;
            int xp2 = x + 2 >= width ? width - 1 : x + 2;
            int sum = src[xm2] + 4 * src[xm1] + 6 * src[x] + 4 * src[xp1] + src[xp2];
            dst[x] = (uint8_t)(sum >> 4);
        }
    }

    #pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        int ym2 = y - 2 < 0 ? 0 : y - 2;
        int ym1 = y - 1 < 0 ? 0 : y - 1;
        int yp1 = y + 1 >= height ? height - 1 : y + 1;
        int yp2 = y + 2 >= height ? height - 1 : y + 2;

        const uint8_t *restrict r0 = temp + (size_t)ym2 * (size_t)width;
        const uint8_t *restrict r1 = temp + (size_t)ym1 * (size_t)width;
        const uint8_t *restrict r2 = temp + (size_t)y   * (size_t)width;
        const uint8_t *restrict r3 = temp + (size_t)yp1 * (size_t)width;
        const uint8_t *restrict r4 = temp + (size_t)yp2 * (size_t)width;
        uint8_t *restrict dst = output + (size_t)y * (size_t)width;

        #pragma omp simd
        for (int x = 0; x < width; x++)
        {
            int sum = r0[x] * 1
                    + r1[x] * 4
                    + r2[x] * 6
                    + r3[x] * 4
                    + r4[x] * 1;
            dst[x] = (uint8_t)(sum >> 4);
        }
    }
}

float magnitude_and_angle(const uint8_t *input, float *output_magnitude, uint8_t *output_angle, int width, int height)
{
    if (!input || !output_magnitude || !output_angle) return 0.0f;

    float max = 0.0f;

    /*SLOWER WAY OF FINDING THE ANGLE USING TAN FUNCTION
    float angle = atan2f(gy, gx) * (180.0f / (float)M_PI);
    if (angle < 0.0f) angle += 180.0f;

    if ((angle >= 0.0f && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f)) output_angle[idx] = 0;
    else if (angle >= 22.5f && angle < 67.5f)  output_angle[idx] = 1;
    else if (angle >= 67.5f && angle < 112.5f) output_angle[idx] = 2;
    else                                       output_angle[idx] = 3;
    */

    /*SLOWER WAY, COLLAPSE(2) + CRITICAL SECTION, THE IF/ELSE CHAIN AND MISSING RESTRICT POINTERS STOP VECTORIZATION
    #pragma omp parallel
    {
        float local_max = 0.0f;

        #pragma omp for collapse(2) nowait
        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                int idx = y * width + x;
                float right_side  = (float)(input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)]);
                float left_side   = (float)(input[(y - 1) * width + (x - 1)] + 2 * input[y * width + (x - 1)] + input[(y + 1) * width + (x - 1)]);
                float gx = right_side - left_side;

                float bottom_side = (float)(input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)]);
                float top_side    = (float)(input[(y - 1) * width + (x - 1)] + 2 * input[(y - 1) * width + x] + input[(y - 1) * width + (x + 1)]);
                float gy = bottom_side - top_side;

                float mag = sqrtf(gx * gx + gy * gy);
                output_magnitude[idx] = mag;
                if (mag > local_max) local_max = mag;

                //FASTER WAY OF FINDING TAN USING GY/GX
                float abs_gx = fabsf(gx);
                float abs_gy = fabsf(gy);
                if (abs_gy <= 0.4142f * abs_gx) output_angle[idx] = 0;
                else if (abs_gy >= 2.4142f * abs_gx) output_angle[idx] = 2;
                else if (gx * gy > 0.0f) output_angle[idx] = 1;
                else output_angle[idx] = 3;
            }
        }

        #pragma omp critical
        {
            if (local_max > max) max = local_max;
        }
    }
    */

    //FASTER WAY, ONE ROW PER THREAD, INTEGER SOBEL SUMS, BRANCHLESS DIRECTION AND AN OPENMP MAX REDUCTION (SIMD)
    #pragma omp parallel for reduction(max:max)
    for (int y = 1; y < height - 1; y++)
    {
        const uint8_t *restrict up   = input + (size_t)(y - 1) * (size_t)width;
        const uint8_t *restrict mid  = input + (size_t)y       * (size_t)width;
        const uint8_t *restrict down = input + (size_t)(y + 1) * (size_t)width;
        float   *restrict mag_row = output_magnitude + (size_t)y * (size_t)width;
        uint8_t *restrict ang_row = output_angle     + (size_t)y * (size_t)width;

        #pragma omp simd reduction(max:max)
        for (int x = 1; x < width - 1; x++)
        {
            int gx = (up[x + 1] + 2 * mid[x + 1] + down[x + 1]) - (up[x - 1] + 2 * mid[x - 1] + down[x - 1]);
            int gy = (down[x - 1] + 2 * down[x] + down[x + 1]) - (up[x - 1] + 2 * up[x] + up[x + 1]);

            float fgx = (float)gx;
            float fgy = (float)gy;
            float mag = sqrtf(fgx * fgx + fgy * fgy);
            mag_row[x] = mag;
            max = mag > max ? mag : max;

            //FASTER WAY OF FINDING TAN USING GY/GX
            float abs_gx = fabsf(fgx);
            float abs_gy = fabsf(fgy);
            ang_row[x] = (uint8_t)(abs_gy <= 0.4142f * abs_gx ? 0
                                 : abs_gy >= 2.4142f * abs_gx ? 2
                                 : gx * gy > 0               ? 1
                                 :                             3);
        }
    }

    return max;
}

void non_maximum_suppression(const float *magnitude_arr, const uint8_t *angles, float *output, int width, int height, float max_mag)
{
    if (!magnitude_arr || !angles || !output) return;

    /*SLOWER WAY, THE SWITCH STOPS VECTORIZATION AND THE NORMALIZATION NEEDS A SECOND PASS OVER THE WHOLE IMAGE
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int idx = y * width + x;
            float mag = magnitude_arr[idx];
            int n1, n2;
            uint8_t dir = angles[idx];

            switch (dir)
            {
            case 0:
                n1 = y * width + (x - 1);
                n2 = y * width + (x + 1);
                break;
            case 1:
                n1 = (y - 1) * width + (x - 1);
                n2 = (y + 1) * width + (x + 1);
                break;
            case 2:
                n1 = (y - 1) * width + x;
                n2 = (y + 1) * width + x;
                break;
            case 3:
                n1 = (y - 1) * width + (x + 1);
                n2 = (y + 1) * width + (x - 1);
                break;
            default:
                n1 = idx;
                n2 = idx;
                break;
            }

            output[idx] = (mag > magnitude_arr[n1] && mag > magnitude_arr[n2]) ? mag : 0.0f;
        }
    }

    if (max_mag > 0.0f) {
        size_t total = (size_t)width * (size_t)height;
        #pragma omp parallel for
        for (size_t i = 0; i < total; i++) {
            output[i] = (output[i] / max_mag) * 255.0f;
        }
    }
    */

    //FASTER WAY, EVERY DIRECTION IS TESTED AND THE RIGHT RESULT IS PICKED WITH BIT MASKS INSTEAD OF BRANCHES (SIMD), NORMALIZATION IS DONE IN THE SAME PASS
    const float scale = max_mag > 0.0f ? 255.0f / max_mag : 0.0f;

    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++)
    {
        const float   *restrict up   = magnitude_arr + (size_t)(y - 1) * (size_t)width;
        const float   *restrict mid  = magnitude_arr + (size_t)y       * (size_t)width;
        const float   *restrict down = magnitude_arr + (size_t)(y + 1) * (size_t)width;
        const uint8_t *restrict dir_row = angles + (size_t)y * (size_t)width;
        float *restrict out = output + (size_t)y * (size_t)width;

        #pragma omp simd
        for (int x = 1; x < width - 1; x++)
        {
            float mag = mid[x];
            int dir = dir_row[x];

            int keep_horizontal = (mag > mid[x - 1])  & (mag > mid[x + 1]);
            int keep_diagonal   = (mag > up[x - 1])   & (mag > down[x + 1]);
            int keep_vertical   = (mag > up[x])       & (mag > down[x]);
            int keep_antidiag   = (mag > up[x + 1])   & (mag > down[x - 1]);

            int keep = ((dir == 0) & keep_horizontal)
                     | ((dir == 1) & keep_diagonal)
                     | ((dir == 2) & keep_vertical)
                     | ((dir == 3) & keep_antidiag);

            out[x] = keep ? mag * scale : 0.0f;
        }
    }
}

void histeresis(const float *nms, uint8_t *output, int width, int height)
{
    if (!nms || !output) return;

    size_t total = (size_t)width * (size_t)height;
    static int *stack = NULL;
    static int *row_count = NULL;
    if (!stack) {
        stack = malloc(total * sizeof(*stack));
        row_count = malloc((size_t)height * sizeof(*row_count));
        if (!stack || !row_count) {
            free(stack);     stack = NULL;
            free(row_count); row_count = NULL;
            return;
        }
    }

    /*SLOWER WAY, THE WHOLE IMAGE IS CLASSIFIED ON A SINGLE THREAD
    int top = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;
            if (nms[idx] >= 20.0f)
            {
                output[idx] = 255;
                stack[top++] = idx;
            }
            else if (nms[idx] >= 8.0f) output[idx] = 128;
            else                       output[idx] = 0;
        }
    }
    */

    //FASTER WAY, ROWS ARE CLASSIFIED IN PARALLEL AND EACH ROW SAVES ITS STRONG PIXELS IN ITS OWN SLICE OF THE STACK
    #pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        const float *restrict src = nms + (size_t)y * (size_t)width;
        uint8_t *restrict dst = output + (size_t)y * (size_t)width;
        int *restrict seeds = stack + (size_t)y * (size_t)width;
        int count = 0;

        for (int x = 0; x < width; x++)
        {
            float v = src[x];
            if (v >= 20.0f)
            {
                dst[x] = 255;
                seeds[count++] = y * width + x;
            }
            else dst[x] = v >= 8.0f ? 128 : 0;
        }
        row_count[y] = count;
    }

    int top = 0;
    for (int y = 0; y < height; y++)
    {
        memmove(stack + top, stack + (size_t)y * (size_t)width, (size_t)row_count[y] * sizeof(*stack));
        top += row_count[y];
    }

    while (top > 0)
    {
        int curr_idx = stack[--top];
        int cx = curr_idx % width;
        int cy = curr_idx / width;
        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                int nx = cx + kx;
                int ny = cy + ky;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                {
                    int neighbor_idx = ny * width + nx;
                    if (output[neighbor_idx] == 128)
                    {
                        output[neighbor_idx] = 255;
                        stack[top++] = neighbor_idx;
                    }
                }
            }
        }
    }

    /*SLOWER WAY, THE CONDITIONAL STORE STOPS VECTORIZATION
    #pragma omp parallel for
    for (size_t i = 0; i < total; i++) {
        if (output[i] == 128) output[i] = 0;
    }
    */

    //FASTER WAY, ONE ROW PER THREAD AND AN UNCONDITIONAL STORE (SIMD)
    #pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        uint8_t *restrict row = output + (size_t)y * (size_t)width;

        #pragma omp simd
        for (int x = 0; x < width; x++) {
            row[x] = row[x] == 255 ? 255 : 0;
        }
    }
}
