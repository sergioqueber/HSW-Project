#include "preprocess.h"
#include "esp_log.h"
#include "model.h"

static const char *TAG = "PREPROCESS";

bool preprocess_image(const uint8_t *raw_image, int8_t *model_input)
{
    ESP_LOGI(TAG, "Preprocessing (Placeholder)...");

    // [TODO]  IMplement preprocessing 
    
    // Fill with zeros for now so it doesn't break inference
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS; i++) {
        model_input[i] = 0;
    }
    
    return true;
}