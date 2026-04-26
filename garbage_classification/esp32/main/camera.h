#ifndef CAMERA_H
#define CAMERA_H

#include <stdint.h>

// Camera definitions for 320x240 RGB565 buffer from camera.cpp
#define FRAME_W 320
#define FRAME_H 240
#define FRAME_C 2 // 2 bytes for RGB565 format

bool camera_init(void);
bool camera_capture_frame(uint8_t *image_buffer);

#endif // CAMERA_H