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
    x_train_list, y_train_list, source_train_list = [], [], []
    x_val_list, y_val_list, source_val_list = [], [], []
    x_test_list, y_test_list, source_test_list = [], [], []

    sources = {
        'web': 0,
        'device': 1
    }

    for source_name, source_id in sources.items():
        source_dir = os.path.join(data_dir, source_name)

        if not os.path.exists(source_dir):
            print(f"Warning: Source directory not found -> {source_dir}")
            continue

        for class_index, class_name in enumerate(CLASSES):
            class_dir = os.path.join(source_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found -> {class_dir}")
                continue

            x, y = _preprocess_directory(class_dir, class_index)

            if len(x) == 0:
                continue

            indices = np.arange(len(x))
            np.random.shuffle(indices)

            num_samples = len(x)
            num_train = int(0.6 * num_samples)
            num_val = int(0.2 * num_samples)

            train_idx = indices[:num_train]
            val_idx = indices[num_train:num_train + num_val]
            test_idx = indices[num_train + num_val:]

            x_train_list.append(x[train_idx])
            y_train_list.append(y[train_idx])
            source_train_list.append(np.full(len(train_idx), source_id, dtype=np.int32))

            x_val_list.append(x[val_idx])
            y_val_list.append(y[val_idx])
            source_val_list.append(np.full(len(val_idx), source_id, dtype=np.int32))

            x_test_list.append(x[test_idx])
            y_test_list.append(y[test_idx])
            source_test_list.append(np.full(len(test_idx), source_id, dtype=np.int32))

    x_train = np.concatenate(x_train_list)
    y_train = np.concatenate(y_train_list)
    source_train = np.concatenate(source_train_list)

    x_val = np.concatenate(x_val_list)
    y_val = np.concatenate(y_val_list)
    source_val = np.concatenate(source_val_list)

    x_test = np.concatenate(x_test_list)
    y_test = np.concatenate(y_test_list)
    source_test = np.concatenate(source_test_list)

    def shuffle_together(x, y, source):
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        return x[indices], y[indices], source[indices]

    x_train, y_train, source_train = shuffle_together(x_train, y_train, source_train)
    x_val, y_val, source_val = shuffle_together(x_val, y_val, source_val)
    x_test, y_test, source_test = shuffle_together(x_test, y_test, source_test)

    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'source_train.npy'), source_train)

    np.save(os.path.join(out_dir, 'x_val.npy'), x_val)
    np.save(os.path.join(out_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(out_dir, 'source_val.npy'), source_val)

    np.save(os.path.join(out_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(out_dir, 'source_test.npy'), source_test)

    print(f"Preprocessing complete! Files saved to {out_dir}")
    print("Train class counts:", np.bincount(y_train, minlength=NUM_CLASSES))
    print("Val class counts:", np.bincount(y_val, minlength=NUM_CLASSES))
    print("Test class counts:", np.bincount(y_test, minlength=NUM_CLASSES))
    print("Train source counts:", np.bincount(source_train, minlength=2))
    print("Val source counts:", np.bincount(source_val, minlength=2))
    print("Test source counts:", np.bincount(source_test, minlength=2))


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
        img_array = img_array.astype(np.float16) / 255.0
    else:
        img_array = img_array.astype(np.float16)

    return img_array

# To run the code, you would use something like this:
if __name__ == '__main__':
    raw_data_path = os.path.abspath('.././HSW-Data')
    output_data_path = os.path.abspath('./garbage_classification/data')
    preprocess_all(raw_data_path, output_data_path)