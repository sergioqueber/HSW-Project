import os
import shutil
import itertools
import numpy as np

# Ensure TensorFlow handles Keras 3 / 2.16 correctly
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras_tuner import BayesianOptimization
from keras_tuner.engine.trial import Trial

from train import preprocess_and_load_data
from preprocess import IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, NUM_CLASSES

def _test_hyperparameters(hyperparameters: dict, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> float:
    # 1. Initial Standard Convolution Layer
    layers = [
        Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)),
        Conv2D(hyperparameters['first_layer_size'], hyperparameters['kernel_size'], padding='same', activation='relu'),
        MaxPooling2D(2),
        Dropout(hyperparameters['dropout_rate'])
    ]
    
    # 2. Middle Separable Convolution Layers (Efficient for ESP32)
    for _ in range(hyperparameters['num_layers']):
        layers.extend([
            SeparableConv2D(hyperparameters['layer_size'], hyperparameters['kernel_size'], padding='same', activation='relu'),
            MaxPooling2D(2),
            Dropout(hyperparameters['dropout_rate'])
        ])
        
    # 3. Output Global Average Pooling and Dense Layer
    # GAP prevents massive parameter explosion in the final layer
    layers.extend([
        GlobalAveragePooling2D(),
        Dropout(hyperparameters['dropout_rate']),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    # Safety check: ensure spatial dimensions don't shrink below 1x1 due to pooling
    num_poolings = 1 + hyperparameters['num_layers']
    if (IMAGE_WIDTH // (2**num_poolings)) < 1:
        print('Invalid architecture (too many pooling layers for image size)', end=' ')
        return 0.0

    # Build and compile model
    model = Sequential(layers)
    model.compile(
        optimizer=Adam(learning_rate=hyperparameters['learning_rate']), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    # Train model with early stopping
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ]
    
    model.fit(
        x_train, y_train, 
        epochs=30, 
        batch_size=hyperparameters['batch_size'], 
        validation_data=(x_val, y_val), 
        callbacks=callbacks, 
        verbose=0
    )

    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    return val_accuracy

def tune_hyperparameters_bayesian(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> dict:
    class MyTuner(BayesianOptimization):
        def run_trial(self, trial: Trial, x_train, y_train, x_val, y_val):
            hp = trial.hyperparameters
            hyperparameters = {
                'learning_rate': hp.Choice('learning_rate', [0.001, 0.0005, 0.0001]),
                'batch_size': hp.Choice('batch_size', [16, 32]), # Keeping low (16 or 32) since 256x256 image arrays consume more RAM
                'dropout_rate': hp.Choice('dropout_rate', [0.1, 0.2, 0.3]),
                'num_layers': hp.Choice('num_layers', [2, 3, 4]),
                'layer_size': hp.Choice('layer_size', [16, 32, 64]),
                'first_layer_size': hp.Choice('first_layer_size', [16, 32]),
                'kernel_size': hp.Choice('kernel_size', [3, 5])
            }
            return _test_hyperparameters(hyperparameters, x_train, y_train, x_val, y_val)

    # Run tuning
    tuner = MyTuner(objective='val_accuracy', max_trials=20, directory='gen', project_name='garbage_tuning')
    tuner.search(x_train, y_train, x_val, y_val)

    # Delete 'gen/garbage_tuning' directory to allow clean re-runs
    shutil.rmtree("gen/garbage_tuning", ignore_errors=True)

    return tuner.get_best_hyperparameters(1)[0].values

if __name__ == '__main__':
    # Load data from the npy files
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_and_load_data()

    print("Starting Bayesian Optimization Hyperparameter Tuning...")
    best_hyperparameters = tune_hyperparameters_bayesian(x_train, y_train, x_val, y_val)

    print('\n======================================')
    print('Best hyperparameters found:')
    print(best_hyperparameters)
    print('======================================')