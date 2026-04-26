import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow_model_optimization as tfmot

from preprocess import preprocess_all, NUM_CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, IMAGE_MEAN, IMAGE_STD
from model import create_model

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../keywords/python')))
from utils.export_tflite import write_model_h_file, write_model_c_file

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


def export_model_to_tflite(model: tf.keras.models.Model, x_train: np.ndarray):
    print("Exporting model to TFLite with Full Integer Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    def representative_dataset():
        for i in range(min(100, len(x_train))):
            yield [x_train[i:i+1].astype(np.float32)]
            
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Enforce pure integer operations matching the ESP32 expectations
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('Input scale:', input_details[0]['quantization'][0])
    print('Input zero point:', input_details[0]['quantization'][1])
    print('Output scale:', output_details[0]['quantization'][0])
    print('Output zero point:', output_details[0]['quantization'][1])

    # Export to ESP32 Headers
    print(f"Exporting C Arrays to {MODEL_C_PATH}...")
    defines = {
        'NUM_CLASSES': NUM_CLASSES,
        'IMAGE_WIDTH': IMAGE_WIDTH,
        'IMAGE_HEIGHT': IMAGE_HEIGHT,
        'CHANNELS': CHANNELS,
        'IMAGE_MEAN': f'{IMAGE_MEAN}f',
        'IMAGE_STD': f'{IMAGE_STD}f',
    }
    
    os.makedirs(os.path.dirname(MODEL_C_PATH), exist_ok=True)
    write_model_h_file(MODEL_H_PATH, defines, [])
    write_model_c_file(MODEL_C_PATH, tflite_model)
    
    os.makedirs(GEN_DIR, exist_ok=True)
    with open(os.path.join(GEN_DIR, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)

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
    # Load the best pre-quantization epoch
    model = load_model('saved_models/best_model.keras')
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # [TO IMPLEMENT] TFLite conversion step
    export_model_to_tflite(model, x_train)
    print("Done. Exported model.c and model.h to ESP32 main folder.")
    
if __name__ == '__main__':
    train()
