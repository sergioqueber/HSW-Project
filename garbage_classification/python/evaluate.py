import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../garbege_classification/python')))
from utils.export_tflite import write_model_h_file, write_model_c_file

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
TFLITE_MODEL_PATH = os.path.join(DATA_DIR, "model.tflite")

def evaluate_tflite_model(tflite_model, x_test, y_test):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    correct = 0

    for i in range(len(x_test)):
        if i % 100 == 0:
            print(f"Evaluating {i}/{len(x_test)}")
        
        x = x_test[i:i+1].astype(np.float32)

        # Quantize float input to int8
        x_quantized = x / input_scale + input_zero_point
        x_quantized = np.clip(x_quantized, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], x_quantized)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        # Dequantize output if needed
        output_float = (output.astype(np.float32) - output_zero_point) * output_scale

        prediction = np.argmax(output_float)
        label = y_test[i]

        if prediction == label:
            correct += 1

    acc = correct / len(x_test)
    print(f"TFLite INT8 Test Accuracy: {acc:.4f}")

x_test = np.load(os.path.join(DATA_DIR, "x_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

with open(TFLITE_MODEL_PATH, "rb") as f:
    tflite_model = f.read()

evaluate_tflite_model(tflite_model, x_test, y_test)

