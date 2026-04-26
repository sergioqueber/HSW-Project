#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "camera.h"
#include "preprocess.h"
#include "inference.h"
#include "model.h"

static const char *TAG = "MAIN";

// Buffer for raw camera captures (320x240 RGB565)
static uint8_t raw_camera_buffer[FRAME_W * FRAME_H * FRAME_C];

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting Garbage Classification...");

    // Initialize Camera hardware
    if (!camera_init()) {
        ESP_LOGE(TAG, "Failed to initialize camera!");
        return;
    }

    // Initialize the TFLite Inference engine
    if (!inference_init()) {
        ESP_LOGE(TAG, "Failed to initialize inference engine!");
        return;
    }

    // Get direct memory pointer to the TFLM input tensor 
    // (This saves us having to copy the memory twice)
    int8_t *model_input_tensor = inference_get_input_tensor();

    // Main inference loop
    while (1) {
        if (camera_capture_frame(raw_camera_buffer)) {
            
            // 1. Process Raw Camera Image into the Neural Net Input shape and constraints
            preprocess_image(raw_camera_buffer, model_input_tensor);

            // 2. Run Inference
            float confidences[NUM_CLASSES] = {0};
            int64_t start_time = esp_timer_get_time();
            
            if (inference_run(confidences)) {
                int64_t end_time = esp_timer_get_time();
                
                // 3. Output results (Find the highest confident prediction)
                int top_class = 0;
                float top_score = 0.0f;
                for (int i = 0; i < NUM_CLASSES; i++) {
                    if (confidences[i] > top_score) {
                        top_score = confidences[i];
                        top_class = i;
                    }
                }
                
                ESP_LOGI(TAG, "Classification: Class %d (%.1f%% confidence) | Inference Time: %lldms", 
                         top_class, top_score * 100.0f, (end_time - start_time) / 1000);
            }
        } else {
             ESP_LOGW(TAG, "Camera capture failed, retrying...");
        }
        
        // Wait a second before classifying the next frame
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}