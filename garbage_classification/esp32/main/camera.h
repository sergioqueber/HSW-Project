#ifndef CAMERA_H
#define CAMERA_H

#include <stdint.h>

// Camera definitions 
#define FRAME_W 400
#define FRAME_H 296
#define FRAME_C 2 // 2 bytes for RGB565 format

bool camera_init(void);
bool camera_capture_frame(uint8_t *image_buffer);

#endif // CAMERA_H