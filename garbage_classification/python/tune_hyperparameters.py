import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gc
import shutil
import numpy as np
import tensorflow as tf
import keras_tuner as kt

from keras.models import Sequential
from keras.layers import (
    Input, SeparableConv2D, MaxPooling2D, Dropout, Dense,
    GlobalAveragePooling2D, BatchNormalization, RandomFlip,
    RandomRotation, RandomZoom, Rescaling
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications import MobileNetV2

from train import preprocess_and_load_data
from preprocess import IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, NUM_CLASSES


# GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"Could not initialize memory growth: {e}")


def make_dataset(x, y, batch_size, weights=None, shuffle=False):
    def gen():
        for i in range(len(x)):
            image = x[i].astype(np.float32)
            label = np.int32(y[i])

            if weights is None:
                yield image, label
            else:
                yield image, label, np.float32(weights[i])

    if weights is None:
        output_signature = (
            tf.TensorSpec(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if shuffle:
        ds = ds.shuffle(1000)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(hp):
    alpha = hp.Choice("mobilenet_alpha", [0.35, 0.5])
    dense_units = hp.Choice("dense_units", [32, 64, 128])
    conv_filters = hp.Choice("conv_filters", [32, 64, 96])
    dropout_rate = hp.Choice("dropout_rate", [0.2, 0.3, 0.4, 0.5])
    learning_rate = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])

    base_model = MobileNetV2(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS),
        alpha=alpha,
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = Sequential([
        Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)),

        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),

        Rescaling(scale=2.0, offset=-1.0),

        base_model,

        SeparableConv2D(conv_filters, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),

        GlobalAveragePooling2D(),
        Dropout(dropout_rate),

        Dense(dense_units, activation="relu"),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


class GarbageTuner(kt.BayesianOptimization):
    def run_trial(self, trial, x_train, y_train, source_train, x_val, y_val):
        tf.keras.backend.clear_session()
        gc.collect()

        hp = trial.hyperparameters

        batch_size = hp.Choice("batch_size", [8, 16])
        device_weight = hp.Choice("device_weight", [1.5, 2.0, 3.0, 5.0])

        sample_weights = np.ones(len(x_train), dtype=np.float32)
        sample_weights[source_train == 1] = device_weight

        train_dataset = make_dataset(
            x_train,
            y_train,
            batch_size=batch_size,
            weights=sample_weights,
            shuffle=True
        )

        val_dataset = make_dataset(
            x_val,
            y_val,
            batch_size=batch_size,
            weights=None,
            shuffle=False
        )

        model = self.hypermodel.build(hp)

        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=4,
                restore_best_weights=True,
                mode="max"
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-5
            )
        ]

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=15,
            callbacks=callbacks,
            verbose=1
        )

        val_accuracy = max(history.history["val_accuracy"])

        self.oracle.update_trial(
            trial.trial_id,
            {"val_accuracy": val_accuracy}
        )

        tf.keras.backend.clear_session()
        del model
        gc.collect()


def main():
    x_train, y_train, source_train, x_val, y_val, x_test, y_test = preprocess_and_load_data()

    print("Train class counts:", np.bincount(y_train, minlength=NUM_CLASSES))
    print("Val class counts:", np.bincount(y_val, minlength=NUM_CLASSES))
    print("Train source counts:", np.bincount(source_train, minlength=2))

    tuner_dir = "gen"
    project_name = "garbage_tuning"

    ## shutil.rmtree(os.path.join(tuner_dir, project_name), ignore_errors=True)

    tuner = GarbageTuner(
        hypermodel=build_model,
        objective=kt.Objective("val_accuracy", direction="max"),
        max_trials=20,
        directory=tuner_dir,
        project_name=project_name,
        overwrite=False
    )

    tuner.search(x_train, y_train, source_train, x_val, y_val)

    best_hp = tuner.get_best_hyperparameters(1)[0]

    print("\n======================================")
    print("Best hyperparameters found:")
    for key, value in best_hp.values.items():
        print(f"{key}: {value}")
    print("======================================")


if __name__ == "__main__":
    main()