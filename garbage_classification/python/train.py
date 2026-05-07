import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow_model_optimization as tfmot

from preprocess import preprocess_all, CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, IMAGE_MEAN, IMAGE_STD, NUM_CLASSES
from model import create_model

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../keywords/python')))
from utils.export_tflite import write_model_h_file, write_model_c_file

# Minimize TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Enable GPU Memory Growth
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"Could not initialize memory growth: {e}")

# Paths and settings
DATA_DIR = '../data/'
GEN_DIR = 'data/'
MODEL_C_PATH = '../esp32/main/model.c'
MODEL_H_PATH = '../esp32/main/model.h'
USE_CACHED_DATA = True  # Set to True to reuse cached preprocessed data, False to force preprocess data

def evaluate_tflite_model(tflite_model, x_test, y_test):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    correct = 0

    for i in range(len(x_test)):
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

    source_train = np.load(GEN_DIR + 'source_train.npy')

    x_val = np.load(GEN_DIR + 'x_val.npy')
    y_val = np.load(GEN_DIR + 'y_val.npy')
    x_test = np.load(GEN_DIR + 'x_test.npy')
    y_test = np.load(GEN_DIR + 'y_test.npy')

    return x_train, y_train, source_train, x_val, y_val, x_test, y_test

def export_model_to_tflite(model, x_train, x_test, y_test):
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
    x_train, y_train, source_train, x_val, y_val, x_test, y_test = preprocess_and_load_data()
    # Build and compile model
    print("Train class counts:", np.bincount(y_train, minlength=NUM_CLASSES))
    print("Val class counts:", np.bincount(y_val, minlength=NUM_CLASSES))
    print("Test class counts:", np.bincount(y_test, minlength=NUM_CLASSES))

    print("Train source counts:", np.bincount(source_train, minlength=2))
    sample_weights = np.ones(len(x_train), dtype=np.float32)
    sample_weights[source_train == 1] = 2.0

    print('Building model...')
    model = create_model()
        
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks for training
    os.makedirs('saved_models', exist_ok=True)
    callbacks = [
        # Reduce learning rate when validation loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint('saved_models/best_model.keras', save_best_only=True, monitor='val_accuracy')
    ]

    # Convert Numpy arrays to tf.data.Dataset to prevent GPU OOM copy errors
    print('Preparing tf.data.Datasets...')
    batch_size = 16  
    # Use generator to prevent tf.data.Dataset from copying the entire array into RAM again
    def gen(x, y, weights=None):
        def _gen():
            for i in range(len(x)):
                image = x[i].astype(np.float32)
                label = np.int32(y[i])

                if weights is None:
                    yield image, label
                else:
                    yield image, label, np.float32(weights[i])
        return _gen

    train_dataset = tf.data.Dataset.from_generator(
        gen(x_train, y_train, sample_weights),
        output_signature=(
            tf.TensorSpec(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_generator(
        gen(x_val, y_val),
        output_signature=(
            tf.TensorSpec(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_generator(
        gen(x_test, y_test),
        output_signature=(
            tf.TensorSpec(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Train model
    print('Training model...')
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=callbacks
    )

    print("Evaluating on test set...")
    # Load the best pre-quantization epoch
    model = load_model('saved_models/best_model.keras')
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    export_model_to_tflite(model, x_train, x_test, y_test)
    print("Done. Exported model.c and model.h to ESP32 main folder.")
    
if __name__ == '__main__':
    train()
