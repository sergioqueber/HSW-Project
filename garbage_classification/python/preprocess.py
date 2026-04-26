import os
import numpy as np
import matplotlib.image as mpimg

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
CHANNELS = 3

CLASSES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass', 
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]
NUM_CLASSES = len(CLASSES)

# Since we're dividing by 255.0, the effective range is [0.0, 1.0].
# For C++ normalization (input - mean) / std:
IMAGE_MEAN = 0.0
IMAGE_STD = 255.0

def preprocess_all(data_dir: str, out_dir: str):
    all_x_data = []
    all_y_data = []

    for class_index, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        
        if os.path.exists(class_dir):
            x, y = _preprocess_directory(class_dir, class_index)
            
            if len(x) > 0:
                all_x_data.append(x)
                all_y_data.append(y)
        else:
            print(f"Warning: Directory not found -> {class_dir}")

    x_all = np.concatenate(all_x_data)
    y_all = np.concatenate(all_y_data)
    
    indices = np.arange(len(x_all))
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]

    num_samples = len(x_all)
    num_train = int(0.6 * num_samples)
    num_val = int(0.2 * num_samples)
    
    x_train = x_all[:num_train]
    y_train = y_all[:num_train]
    x_val = x_all[num_train:num_train + num_val]
    y_val = y_all[num_train:num_train + num_val]
    x_test = x_all[num_train + num_val:]
    y_test = y_all[num_train + num_val:]

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'x_val.npy'), x_val)
    np.save(os.path.join(out_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(out_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)

    del all_x_data, all_y_data, x_all, y_all
    print(f"Preprocessing complete! Files saved to {out_dir}")


def _preprocess_directory(data_dir: str, class_index: int) -> tuple[np.ndarray, np.ndarray]:
    print('Preprocessing directory: ', data_dir)
    images = []   
    valid_extensions = ('.jpg', '.png')

    for img_file in os.listdir(data_dir):
        if img_file.lower().endswith(valid_extensions):
            file_path = os.path.join(data_dir, img_file)
            
            try:
                img_array = mpimg.imread(file_path)
                processed_img = preprocess_image(img_array)
                images.append(processed_img)
                
            except Exception as e:
                print(f"Warning: Could not process {img_file}. Error: {e}")

    if not images:
         print(f"No valid images found in {data_dir}.")
         return np.array([]), np.array([])

    return np.stack(images), np.full(len(images), class_index)


def preprocess_image(img_array: np.ndarray) -> np.ndarray:
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    original_height, original_width = img_array.shape[:2]
    row_indices = np.linspace(0, original_height - 1, IMAGE_HEIGHT).astype(int)
    col_indices = np.linspace(0, original_width - 1, IMAGE_WIDTH).astype(int)
    
    img_array = img_array[row_indices, :][:, col_indices]

    if img_array.max() > 1.0:
        img_array = img_array.astype(np.float32) / 255.0
    else:
        img_array = img_array.astype(np.float32)

    return img_array

# To run the code, you would use something like this:
if __name__ == '__main__':
    raw_data_path = os.path.abspath('.././HSW-Data')
    output_data_path = os.path.abspath('./garbage_classification/data')
    preprocess_all(raw_data_path, output_data_path)