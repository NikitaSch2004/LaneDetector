#ifndef LANE_ASSIST_H
#define LANE_ASSIST_H

#include <stdint.h>

void roi_mask(uint8_t *input, int width, int height);
void hough_transform(const uint8_t *input, unsigned char *draw, int width, int height);

#endif // LANE_ASSIST_H