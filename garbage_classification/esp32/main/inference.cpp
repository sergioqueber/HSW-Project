#include "inference.h"
#include "model.h" 
#include "esp_log.h"

// TFLite Micro Headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "INFERENCE";

// TFLM Globals
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor Arena (Memory pool for the model computations)
// You may need to tune this size down or up depending on your model size!
constexpr int kTensorArenaSize = 300 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

bool inference_init(void)
{
    ESP_LOGI(TAG, "Initializing TFLite Micro...");

    // Get the model (model_binary is populated by Python export script)
    tflite_model = tflite::GetModel(model_binary);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model provided is schema version %ld not equal to supported version %d.",
                 tflite_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Set up the ops resolver
    // We register the exact operations that our python model (SeparableConv2D, Dense, Softmax) uses.
    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddPadding(); // Oftentimes injected during SeparableConv2D

    // Build an interpreter to run the model
    static tflite::MicroInterpreter static_interpreter(
        tflite_model, resolver, tensor_arena, kTensorArenaSize);
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

    return true;
}

int8_t* inference_get_input_tensor(void)
{
    return input->data.int8;
}

bool inference_run(float* confidences)
{
    // Run the model on the preprocessed input
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed.");
        return false;
    }

    // Dequantize outputs back into readable percentage probabilities (0.0 to 1.0 float)
    for (int i = 0; i < NUM_CLASSES; i++) {
        int8_t output_val = output->data.int8[i];
        float probability = (output_val - output->params.zero_point) * output->params.scale;
        confidences[i] = probability;
    }

    return true;
}