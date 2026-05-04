import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

from preprocess import preprocess_image, CLASSES

MODEL_PATH = "saved_models/best_model.keras"
DEFAULT_IMAGE_PATH = "test_image.jpg"


def load_trained_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")
    return load_model(model_path)


def load_image_as_numpy(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def predict_single_image(model, image_path: str):
    print(f"Predicting image: {image_path}")

    raw_image = load_image_as_numpy(image_path)

    # Use the exact same preprocessing as training
    input_data = preprocess_image(raw_image)

    # Add batch dimension: (256, 256, 3) -> (1, 256, 256, 3)
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

    predictions = model.predict(input_data, verbose=0)

    predicted_class_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class_index])

    predicted_label = CLASSES[predicted_class_index]

    print()
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print()

    print("All class scores:")
    for class_name, score in zip(CLASSES, predictions[0]):
        print(f"{class_name}: {score:.4f}")

    return predicted_label, confidence


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH

    try:
        model = load_trained_model(MODEL_PATH)
        predict_single_image(model, image_path)

    except Exception as e:
        print(f"Error during prediction: {e}")