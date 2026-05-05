#include "inference.h"
#include "model.h" 
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h" 

// TFLite Micro Headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "INFERENCE";

// TFLM Globals
static const tflite::Model* tflite_model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

static const char* CLASS_NAMES[] = {
    "battery", "biological", "cardboard", "clothes", "glass", 
    "metal", "paper", "plastic", "shoes", "trash"
};

// Tensor Arena (Memory pool for the model computations)
// MobileNetV2 has much larger intermediate feature maps than simple CNNs.
// A 256x256 image passing through MobileNet will need ~1.5MB - 2MB of working RAM.
// Since internal SRAM on the ESP32 is only ~520KB, this MUST be allocated in external PSRAM (SPIRAM).
// TFLite Micro requires 16-byte alignment for the tensor arena to avoid corruption.
constexpr int kTensorArenaSize = 2 * 1024 * 1024;
constexpr int kAlignmentBytes = 16;
static uint8_t* tensor_arena = nullptr;
static uint8_t* tensor_arena_raw = nullptr;  // Unaligned allocation for cleanup

bool inference_init(void)
{
    ESP_LOGI(TAG, "Initializing TFLite Micro...");

    if (tensor_arena == nullptr) {
        ESP_LOGI(TAG, "Allocating %d bytes for Tensor Arena in PSRAM with 16-byte alignment...", kTensorArenaSize);
        // Allocate extra bytes to ensure we can align to 16-byte boundary
        tensor_arena_raw = (uint8_t*) heap_caps_malloc(kTensorArenaSize + kAlignmentBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (tensor_arena_raw == nullptr) {
            ESP_LOGE(TAG, "Failed to allocate tensor arena in PSRAM. Check if PSRAM is enabled in sdkconfig!");
            return false;
        }
        // Align to 16-byte boundary
        uintptr_t addr = (uintptr_t)tensor_arena_raw;
        uintptr_t aligned_addr = (addr + (kAlignmentBytes - 1)) & ~(kAlignmentBytes - 1);
        tensor_arena = (uint8_t*)aligned_addr;
    }

    // Get the model (model_binary is populated by Python export script)
    tflite_model = tflite::GetModel(model_binary);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model provided is schema version %ld not equal to supported version %d.",
                 tflite_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Set up the ops resolver
    // MobileNetV2 uses many complex ops (Add, Mean, Relu6, etc), we map the majority here:
    static tflite::MicroMutableOpResolver<25> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddSub();
    resolver.AddMean();
    resolver.AddPad();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddLogistic();

    // Build an interpreter to run the model
    static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return false;
    }

    // Get information about the memory area to use for the model's input
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    ESP_LOGI(TAG, "Input tensor dimensions: %d [%d, %d, %d, %d], type: %d",
             (int)input->dims->size, 
             (int)input->dims->data[0], (int)input->dims->data[1], 
             (int)input->dims->data[2], (int)input->dims->data[3], (int)input->type);

    // Verify input tensor dimensions
    if (input->dims->size != 4 || 
        input->dims->data[1] != 256 || input->dims->data[2] != 256 ||
        input->dims->data[3] != 3) {
        ESP_LOGE(TAG, "Bad input tensor parameters in model");
        return false;
    }

    ESP_LOGI(TAG, "Inference engine initialized successfully");
    return true;
}

int8_t* inference_get_input_tensor(void)
{
    return input->data.int8;
}

float inference_get_input_scale(void)
{
    return input->params.scale;
}

int inference_get_input_zero_point(void)
{
    return input->params.zero_point;
}

bool inference_run(void)
{
    if (!interpreter || !input || !output) {
        ESP_LOGE(TAG, "Interpreter or model input/output is null");
        return false;
    }

    // Run the model on this input and make sure it succeeds.
    long long start_time = esp_timer_get_time();
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed.");
        return false;
    }
    long long end_time = esp_timer_get_time();
    long long inference_time = (end_time - start_time) / 1000;
    
    // Process the output
    float max_prob = 0;
    int max_idx = -1;
    int num_classes = output->dims->data[1];

    for (int i = 0; i < num_classes; i++) {
        // Output might be int8 if quantized, or float
        float probability;
        if (output->type == kTfLiteInt8) {
            int8_t output_val = output->data.int8[i];
            probability = (output_val - output->params.zero_point) * output->params.scale;
        } else {
            probability = output->data.f[i];
        }
        
        if (probability > max_prob) {
            max_prob = probability;
            max_idx = i;
        }
    }

    ESP_LOGI(TAG, "Classification: %s (%.1f%% confidence) | Inference Time: %lldms", 
             (max_idx >= 0 && max_idx < 10) ? CLASS_NAMES[max_idx] : "Unknown", 
             max_prob * 100, 
             inference_time);
    return true;
}