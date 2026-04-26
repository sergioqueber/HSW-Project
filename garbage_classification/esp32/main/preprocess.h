#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <stdint.h>

// Preprocesses a raw RGB565 320x240 image into the model input tensor (120x120x3 int8).
bool preprocess_image(const uint8_t *raw_image, int8_t *model_input);

#endif // PREPROCESS_H