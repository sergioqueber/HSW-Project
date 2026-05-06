import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

from preprocess import preprocess_image, CLASSES

MODEL_PATH = "data/model.tflite"
DEFAULT_IMAGE_PATH = "test_image.jpg"


def load_tflite_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading TFLite model from: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def predict_single_image_tflite(interpreter, image_path: str):
    print(f"Predicting image: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    raw_image = np.array(img)
    
    # Preprocess
    input_data = preprocess_image(raw_image)
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Handle quantized input (INT8)
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.int8:
        # Apply quantization
        scale = input_details[0]['quantization'][0]  # scale
        zero_point = input_details[0]['quantization'][1]  # zero point
        input_data = (input_data / scale + zero_point).astype(np.int8)
    
    # Set input and invoke
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get output
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Handle quantized output (INT8)
    output_dtype = output_details[0]['dtype']
    if output_dtype == np.int8:
        # Dequantize output
        scale = output_details[0]['quantization'][0]
        zero_point = output_details[0]['quantization'][1]
        predictions = (predictions.astype(np.float32) - zero_point) * scale
    
    predicted_class_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class_index])
    predicted_label = CLASSES[predicted_class_index]
    
    print(f"\nPrediction: {predicted_label}")
    print(f"Confidence: {confidence:.4f}\n")
    
    print("All class scores:")
    for class_name, score in zip(CLASSES, predictions[0]):
        print(f"{class_name}: {score:.4f}")
    
    return predicted_label, confidence


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    
    try:
        interpreter = load_tflite_model(MODEL_PATH)
        predict_single_image_tflite(interpreter, image_path)
    except Exception as e:
        print(f"Error during prediction: {e}")