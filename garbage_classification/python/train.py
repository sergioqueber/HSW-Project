import os
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow_model_optimization as tfmot # For Quantization Aware Training

from preprocess import preprocess_all 
from model import create_model

# Minimize TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Paths and settings
DATA_DIR = '../data/'
GEN_DIR = 'gen/'
MODEL_C_PATH = '../esp32/main/model.c'
MODEL_H_PATH = '../esp32/main/model.h'
USE_CACHED_DATA = True  # Set to True to reuse cached preprocessed data, False to force preprocess data

def preprocess_and_load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data using preprocess.py routines.
    """
    # Preprocess data if not done already
    if not USE_CACHED_DATA \
            or not os.path.exists(GEN_DIR + 'x_train.npy') or not os.path.exists(GEN_DIR + 'y_train.npy') \
            or not os.path.exists(GEN_DIR + 'x_val.npy') or not os.path.exists(GEN_DIR + 'y_val.npy') \
            or not os.path.exists(GEN_DIR + 'x_test.npy') or not os.path.exists(GEN_DIR + 'y_test.npy'):
        preprocess_all(DATA_DIR, GEN_DIR)

    # Load preprocessed data
    x_train = np.load(GEN_DIR + 'x_train.npy')
    y_train = np.load(GEN_DIR + 'y_train.npy')
    x_val = np.load(GEN_DIR + 'x_val.npy')
    y_val = np.load(GEN_DIR + 'y_val.npy')
    x_test = np.load(GEN_DIR + 'x_test.npy')
    y_test = np.load(GEN_DIR + 'y_test.npy')

    return x_train, y_train, x_val, y_val, x_test, y_test


def train():
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_and_load_data()

    # Build and compile model
    print('Building model...')
    model = create_model()
    
    # [TO IMPLEMENT] Post-Training Quantization (PTQ) vs Quantization Aware Training (QAT)
    # If using QAT, you would wrap the model here:
    # quantize_model = tfmot.quantization.keras.quantize_model
    # model = quantize_model(model)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks for training
    os.makedirs('saved_models', exist_ok=True)
    callbacks = [
        # [TO IMPLEMENT] Tune EarlyStopping patience
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('saved_models/best_model.keras', save_best_only=True, monitor='val_accuracy')
    ]

    # Train model
    print('Training model...')
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=callbacks
    )

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # [TO IMPLEMENT] TFLite conversion step
    # After training, convert to TFLite with integer quantization:
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # # Provide representative dataset generator for full integer quantization
    # tflite_model = converter.convert()
    
    # [TO IMPLEMENT] Write to C array using logic from `utils/export_tflite.py`
    # write_model_c_file(MODEL_C_PATH, tflite_model)
    
if __name__ == '__main__':
    train()
