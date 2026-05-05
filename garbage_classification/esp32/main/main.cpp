#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_camera.h"
#include "driver/usb_serial_jtag.h"

#include <cstring>

#include "camera.h"
#include "preprocess.h"
#include "inference.h"
#include "model.h"

static const char *TAG = "MAIN";

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting Garbage Classification...");

    // Initialize the camera
    if (!camera_init()) {
        ESP_LOGE(TAG, "Failed to initialize camera");
        return;
    }
    ESP_LOGI(TAG, "Camera initialized");

    // Initialize the inference engine
    if (!inference_init()) {
        ESP_LOGE(TAG, "Failed to initialize inference engine!");
        return;
    }

    // Get direct memory pointer to the TFLM input tensor 
    // (This saves us having to copy the memory twice)
    int8_t *model_input_tensor = inference_get_input_tensor();
    float input_scale = inference_get_input_scale();
    int input_zero_point = inference_get_input_zero_point();

    while (1) {
        // 1. Capture a frame from the camera
        camera_fb_t *fb = camera_capture_frame();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        camera_return_frame(fb);

        // 2. Preprocess the image
        ESP_LOGI(TAG, "Preprocessing image...");
        if (preprocess_image(fb, model_input_tensor, input_scale, input_zero_point) != ESP_OK) {
            ESP_LOGE(TAG, "Image preprocessing failed");
            esp_camera_fb_return(fb); // Return frame buffer
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }
        
        // Return the frame buffer as soon as we're done with it
        esp_camera_fb_return(fb);
        ESP_LOGI(TAG, "Image preprocessed and buffer returned");

        // 3. Run inference and yield to prevent watchdog timeout
        ESP_LOGI(TAG, "Running inference...");
        inference_run();

        // Add a small delay to allow other tasks to run, preventing watchdog timeouts
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}