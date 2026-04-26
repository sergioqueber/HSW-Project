import os
import numpy as np

# Preprocessing parameters
NUM_CLASSES = 6 # E.g., paper, plastic, glass, metal, organic, e-waste
IMAGE_WIDTH = 120 
IMAGE_HEIGHT = 120 
CHANNELS = 3

IMAGE_MEAN = 127.5 # Values to be calculated based on dataset
IMAGE_STD = 127.5

def preprocess_all(data_dir: str, out_dir: str):
    # Load and preprocess all directories
    # [TO IMPLEMENT] Iterate classes and preprocess image directory

    # [TO IMPLEMENT] x_all = np.concatenate([class1_x, class2_x, ...])
    # [TO IMPLEMENT] y_all = np.concatenate([class1_y, class2_y, ...])
    
    # [TO IMPLEMENT] Shuffle data
    
    # [TO IMPLEMENT] Split train/val/test
    pass

def _preprocess_directory(data_dir: str, class_index: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess all image files in a directory for a specific class.
    :param data_dir: Path to the directory containing image files.
    :param class_index: Class index to be assigned to all files in this directory.
    :return: All preprocessed images and their corresponding labels as numpy arrays.
    """
    print('Preprocessing directory: ', data_dir)
    images = []
    
    # [TO IMPLEMENT: Iterate files, load images, resize/crop for camera FOV]
    # [TO IMPLEMENT: Normalize images, handle grayscale vs color]

    return np.stack(images), np.full(len(images), class_index)


def preprocess_image(image_data: np.ndarray) -> np.ndarray:
    """
    Preprocess raw image data into model input format.
    :param image_data: Raw image data as a numpy array.
    :return: Preprocessed image as a numpy array of shape (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS).
    """
    # [TO IMPLEMENT: Resize/cropping logic to square or specific aspect ratio]
    # [TO IMPLEMENT: Standardization e.g., (image_data - IMAGE_MEAN) / IMAGE_STD]
    return image_data
