#ifndef CAMERA_H
#define CAMERA_H

#include <stdint.h>

// Camera definitions 
#define FRAME_W 400
#define FRAME_H 296
#define FRAME_C 2 // 2 bytes for RGB565 format

bool camera_init(void);
camera_fb_t* camera_capture_frame(void);
void camera_return_frame(camera_fb_t* fb);

#endif // CAMERA_H