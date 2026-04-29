import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Dropout, Dense, GlobalAveragePooling2D, BatchNormalization, RandomFlip, RandomRotation, RandomZoom, Rescaling
from keras.applications import MobileNetV2

from preprocess import CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, NUM_CLASSES

def create_model() -> tf.keras.Model:
    """
    Define a CNN combining Transfer Learning (MobileNetV2) and custom Convolutional layers.
    Strictly optimized to perfectly fit and run smoothly on the ESP32 microcontroller.
    """
    # Load MobileNetV2 with alpha=0.35. 
    # This massively reduces parameters to ensure the INT8 Quantized model is < 500KB!
    base_model = MobileNetV2(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS),
        alpha=0.35,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model to retain its learned feature extraction
    base_model.trainable = False

    model = Sequential([
        Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)),
        
        # Data Augmentation (Automatically bypassed by TFLite during inference)
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        
        # MobileNet expects pixels mapped to [-1, 1]
        Rescaling(scale=2.0, offset=-1.0),
        
        # 1. Base Model Feature Extractor (Alpha 0.35 is super light)
        base_model,
        
        # 2. Microcontroller-friendly Separable Conv
        # We replace the massive Conv2D(512) with a tiny SeparableConv2D(64).
        # Normal Conv2D multiplies everything heavily; Separable splits spatial and depth, saving 90% parameters.
        SeparableConv2D(64, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        
        # 3. Global Feature Extraction & Classification Head
        GlobalAveragePooling2D(),
        Dropout(0.4),
        
        # Small Dense brain (64 neurons instead of 256/512)
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(NUM_CLASSES, activation='softmax') 
    ])
    return model
