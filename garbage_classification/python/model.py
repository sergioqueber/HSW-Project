import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Dropout, Flatten, Dense

from preprocess import NUM_CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS

def create_model() -> keras.models.Model:
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
        
        SeparableConv2D(32, kernel_size=3, padding='same', activation='relu'), 
        MaxPooling2D(pool_size=2),  
        Dropout(0.1),
        
        Flatten(), 
        Dense(NUM_CLASSES, activation='softmax') 
    ])
    return model
