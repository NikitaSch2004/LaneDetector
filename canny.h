#ifndef CANNY_H
#define CANNY_H

#include <stdint.h>

void rgbToGrayscale(const uint8_t *img, uint8_t *output, int width, int height, int channels);
void cannyEdge(const uint8_t *img, uint8_t *output, int width, int height);
void gaussian_blur_conv(const uint8_t *img, uint8_t *output, int width, int height);
float magnitude_and_angle(const uint8_t *input, float *output_magnitude, uint8_t *output_angle, int width, int height);
void non_maximum_suppression(const float *magnitude_arr, const uint8_t *angles, float *output, int width, int height, float max_mag);
void histeresis(const float *nms, uint8_t *output, int width, int height);

#endif // CANNY_H