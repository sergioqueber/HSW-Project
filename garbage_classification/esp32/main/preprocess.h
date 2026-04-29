#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "esp_camera.h"
#include "esp_err.h"

/**
 * @brief Preprocesses the captured image for the model.
 * 
 * This function takes the camera frame buffer containing a JPEG image,
 * decodes it, and converts it into a 3-channel RGB int8 tensor
 * for the quantized model. The pixel values are normalized and quantized.
 * 
 * @param fb Pointer to the camera frame buffer (must be JPEG format).
 * @param out_buffer Pointer to the int8 buffer where the preprocessed data will be stored.
 * @param scale The quantization scale of the input tensor.
 * @param zero_point The quantization zero point of the input tensor.
 * @return ESP_OK on success, ESP_FAIL on failure.
 */
esp_err_t preprocess_image(camera_fb_t *fb, int8_t* out_buffer, float scale, int zero_point);

#endif // PREPROCESS_H