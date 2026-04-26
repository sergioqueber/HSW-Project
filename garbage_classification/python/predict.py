import os
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Assuming preprocess_image from preprocess.py scales and resizes the image correctly
from preprocess import preprocess_image, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS

# Example: Mapping of class index to class name
CLASS_LABELS = {
    0: "Paper",
    1: "Plastic",
    2: "Glass",
    3: "Metal",
    4: "Organic",
    5: "E-Waste"
}

def load_trained_model(model_path: str):
    """
    Load a saved .keras or .h5 model.
    If you are loading a quantized .tflite model, you'll need the TFLite Interpreter instead.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    return model

def load_tflite_model(tflite_path: str):
    """
    Loads a TFLite model to verify the performance after quantization.
    This is crucial to ensure the quantization process hasn't tanked accuracy.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_single_image(model, image_path: str):
    """
    Predicts the class for a single test image.
    """
    print(f"Predicting image: {image_path}")
    
    # [TO IMPLEMENT] Load the image using OpenCV or PIL
    # img = cv2.imread(image_path)
    
    # Placeholder for un-processed raw image data
    # Replace this with the actual image array loaded above
    raw_image_data = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)) 
    
    # 1. Preprocess exactly as done during training
    input_data = preprocess_image(raw_image_data)
    
    # 2. Add batch dimension (model expects batch size as first dimension)
    input_data = np.expand_dims(input_data, axis=0)
    
    # 3. Inference
    predictions = model.predict(input_data)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_index]
    
    predicted_label = CLASS_LABELS.get(predicted_class_index, "Unknown")
    
    print(f"Prediction: {predicted_label} (Confidence: {confidence:.2f})")
    return predicted_label, confidence

def evaluate_test_set(model, test_images_dir: str):
    """
    Evaluate the model on a generic directory to get confusion matrices.
    """
    # [TO IMPLEMENT] Iteratively score an entire dataset directory or 
    # use the previously generated x_test.npy array to print precision, recall, f1 
    # using utils/eval_utils.py (compute_precision_recall_f1, print_confusion_matrix)
    pass

if __name__ == '__main__':
    # Default model path saved from train.py
    model_file = 'saved_models/best_model.keras'
    
    # [TO IMPLEMENT] Provide a CLI or hardcode an image for quick tests
    test_image_file = 'test_image.jpg'
    
    try:
        model = load_trained_model(model_file)
        if os.path.exists(test_image_file):
            predict_single_image(model, test_image_file)
        else:
            print("No test image found to predict.")
    except Exception as e:
        print(f"Error during prediction: {e}")