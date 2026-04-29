#include "preprocess.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

static const char *TAG = "PREPROCESS";

// Defined by the model input
const int kInputWidth = 256;
const int kInputHeight = 256;
const int kInputChannels = 3;

esp_err_t preprocess_image(camera_fb_t *fb, int8_t* out_buffer, float scale, int zero_point) {
    if (!fb || fb->format != PIXFORMAT_RGB565) {
        ESP_LOGE(TAG, "Invalid frame buffer or format is not RGB565. Got format: %d", fb ? fb->format : -1);
        return ESP_FAIL;
    }
    if (!out_buffer) {
        ESP_LOGE(TAG, "Invalid output buffer");
        return ESP_FAIL;
    }

    uint16_t *img_data = (uint16_t *)fb->buf;

    // Calculate crop offset to center the 240x240 crop from the 320x240 frame
    int offset_x = (fb->width - kInputWidth) / 2;
    int offset_y = (fb->height - kInputHeight) / 2;

    int out_idx = 0;
    
    // Loop through the cropped region
    for (int y = 0; y < kInputHeight; y++) {
        for (int x = 0; x < kInputWidth; x++) {
            // Source index in the full RGB565 frame
            int src_x = x + offset_x;
            int src_y = y + offset_y;
            int src_idx = src_y * fb->width + src_x;

            uint16_t pixel = img_data[src_idx];

            // Extract RGB values (0-255 scale) from RGB565 and divide by 255.0
            float r = ((pixel & 0xF800) >> 8) / 255.0f;
            float g = ((pixel & 0x07E0) >> 3) / 255.0f;
            float b = ((pixel & 0x001F) << 3) / 255.0f;

            // Quantize: q = val / scale + zero_point
            int32_t r_q = (int32_t)(r / scale) + zero_point;
            int32_t g_q = (int32_t)(g / scale) + zero_point;
            int32_t b_q = (int32_t)(b / scale) + zero_point;

            // Clamp to int8 bounds and store
            out_buffer[out_idx++] = (int8_t)((r_q < -128) ? -128 : ((r_q > 127) ? 127 : r_q));
            out_buffer[out_idx++] = (int8_t)((g_q < -128) ? -128 : ((g_q > 127) ? 127 : g_q));
            out_buffer[out_idx++] = (int8_t)((b_q < -128) ? -128 : ((b_q > 127) ? 127 : b_q));
        }
    }

    ESP_LOGI(TAG, "Image preprocessed successfully (RGB565 crop -> Quantized Int8)");
    return ESP_OK;
}