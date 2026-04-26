import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D

from preprocess import CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, NUM_CLASSES

def create_model() -> tf.keras.Model:
    """
    Define the Convolutional Neural Network architecture.
    This model should be lightweight enough to run on the ESP32.
    """
    model = Sequential([
        Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)),
        
        # Initial standard Conv2D to extract basic features (edges/colors)
        Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu'), 
        MaxPooling2D(pool_size=2),  
        Dropout(0.1),
        
        # Switch to SeparableConv2D for parameter efficiency
        SeparableConv2D(32, kernel_size=3, padding='same', activation='relu'), 
        MaxPooling2D(pool_size=2),  
        Dropout(0.1),
        
        SeparableConv2D(64, kernel_size=3, padding='same', activation='relu'), 
        MaxPooling2D(pool_size=2),  
        
        # Use GlobalAveragePooling2D instead of Flatten()
        # Flattening an 8x8x64 tensor creates 4,000+ features. 
        # GlobalAvgPool averages the spatial dimensions, leaving just 64 features.
        GlobalAveragePooling2D(), 
        Dropout(0.2),
        Dense(NUM_CLASSES, activation='softmax') 
    ])
    return model
