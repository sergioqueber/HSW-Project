import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from preprocess import NUM_CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS

def create_model() -> keras.models.Model:
    """
    Define the Convolutional Neural Network architecture.
    This model should be lightweight enough to run on the ESP32.
    """
    model = Sequential([
        Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)),
        
        # [TO IMPLEMENT] Tune filters, kernel sizes, and stride for optimization
        # E.g., DepthwiseConv2D might be better for microcontrollers
        Conv2D(16, 3, activation='relu'), 
        MaxPooling2D(2),  
        Dropout(0.1),
        
        Conv2D(32, 3, activation='relu'), 
        MaxPooling2D(2),  
        Dropout(0.1),
        
        Conv2D(32, 3, activation='relu'), 
        MaxPooling2D(2),  
        Dropout(0.1),
        
        Flatten(), 
        Dense(NUM_CLASSES, activation='softmax') 
    ])
    return model
