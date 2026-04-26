# Garbage Classification Model Training

This directory contains the Python scripts to process the image dataset, train the garbage classification model, and export it for the ESP32.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your dataset to a `dataset/` directory.

## Workflow

1. Preprocess raw data (crop, normalize) in `preprocess.py`
2. Define your neural network architecture in `model.py`
3. Train the model and optimize export in `train.py`
4. Use `predict.py` to benchmark tests on new single images against the exported TFLite.
