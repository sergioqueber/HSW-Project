// ESP includes
#include "esp_camera.h"
#include "esp_log.h"
#include <string.h>

// Project includes
#include "camera.h"

static const char *TAG = "CAMERA";

static camera_config_t get_camera_config()
{
    camera_config_t config = {};

    // LEDC (for XCLK)
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;

    // Data pins
    config.pin_d0 = 15; // Y2_GPIO_NUM
    config.pin_d1 = 17; // Y3_GPIO_NUM
    config.pin_d2 = 18; // Y4_GPIO_NUM
    config.pin_d3 = 16; // Y5_GPIO_NUM
    config.pin_d4 = 14; // Y6_GPIO_NUM
    config.pin_d5 = 12; // Y7_GPIO_NUM
    config.pin_d6 = 11; // Y8_GPIO_NUM
    config.pin_d7 = 48; // Y9_GPIO_NUM

    // Control pins
    config.pin_xclk  = 10; // XCLK_GPIO_NUM
    config.pin_pclk  = 13; // PCLK_GPIO_NUM
    config.pin_vsync = 38; // VSYNC_GPIO_NUM
    config.pin_href  = 47; // HREF_GPIO_NUM

    // SCCB (I2C) pins
    config.pin_sccb_sda = 40; // SIOD_GPIO_NUM
    config.pin_sccb_scl = 39; // SIOC_GPIO_NUM

    // Power-down and reset (not used on XIAO)
    config.pin_pwdn  = -1; // PWDN_GPIO_NUM
    config.pin_reset = -1; // RESET_GPIO_NUM

    // Clock
    config.xclk_freq_hz = 20000000;
    // Frame buffer settings
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count    = 2;
    config.grab_mode   = CAMERA_GRAB_LATEST;

    // Pixel format: RGB565 is efficient to convert to RGB888
    config.pixel_format = PIXFORMAT_RGB565;

    // Resolution: CIF (400x296) so we can crop 256x256
    config.frame_size = FRAMESIZE_CIF;

    // JPEG-specific fields (ignored for RGB565 but set to sane defaults)
    config.jpeg_quality = 12;

    return config;
}

/**
 * Initialize the camera.
 * @return true on success, false on failure.
 */
bool camera_init(void)
{
    ESP_LOGI(TAG, "Initializing camera...");

    camera_config_t camera_config = get_camera_config();
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x.", err);
        return false;
    }

    ESP_LOGI(TAG, "Camera initialized: %dx%d RGB565.", FRAME_W, FRAME_H);

    ESP_LOGI(TAG, "Waiting for AE/AWB to stabilise...");
    for (int i = 0; i < 10; i++) {
        camera_fb_t *warmup = esp_camera_fb_get();
        if (warmup) {
            esp_camera_fb_return(warmup);
        }
    }

    // By default the sensor does NOT apply gamma correction when outputting RGB565,
    // producing near-linear-light values. Training images are sRGB JPEG files
    // (gamma-corrected), so without this the model sees a systematically different
    // colour distribution from what it was trained on.
    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor) {
        sensor->set_raw_gma(sensor, 1); 
        sensor->set_lenc(sensor, 1);     
        sensor->set_gainceiling(sensor, (gainceiling_t)1); 
        sensor->set_awb_gain(sensor, 1); 
        sensor->set_exposure_ctrl(sensor, 1);
        sensor->set_aec2(sensor, 1);   
    } else {
        ESP_LOGW(TAG, "Could not get sensor handle for ISP configuration");
    }

    ESP_LOGI(TAG, "Camera ready.");
    
    return true;
}

/**
 * Capture a frame.
 * @param image_buffer Pointer to the output buffer (must be at least FRAME_W x FRAME_H x FRAME_C bytes).
 * @return true on success, false on failure.
 */
camera_fb_t* camera_capture_frame(void)
{
    // Capture a frame from the camera
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Failed to get frame buffer.");
        return nullptr;
    }

    ESP_LOGI(TAG, "Captured frame: %dx%d, len=%d, format=%d",
             fb->width, fb->height, fb->len, fb->format);

    // Verify frame size and format
    if (fb->width != FRAME_W || fb->height != FRAME_H || fb->format != PIXFORMAT_RGB565) {
        ESP_LOGE(TAG, "Unexpected frame format: %dx%d, format %d.", fb->width, fb->height, fb->format);
        esp_camera_fb_return(fb);
        return nullptr;
    }

    // Verify buffer size
    if (fb->len < FRAME_W * FRAME_H * FRAME_C) {
        ESP_LOGE(TAG, "Frame buffer too small: %d bytes, expected at least %d bytes.", fb->len, FRAME_W * FRAME_H * FRAME_C);
        esp_camera_fb_return(fb);
        return nullptr;
    }

    size_t expected_min_len = fb->width * fb->height * 2; // RGB565 = 2 bytes/pixel

    if (fb->len < expected_min_len) {
        ESP_LOGE(TAG, "Frame buffer too small: len=%d, expected at least=%d",
                 fb->len, expected_min_len);
        esp_camera_fb_return(fb);
        return nullptr;
    }

    return fb;
}

void camera_return_frame(camera_fb_t* fb)
{
    if (fb) {
        esp_camera_fb_return(fb);
    }
}