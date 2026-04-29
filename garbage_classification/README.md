# Garbage Classification

This directory contains the garbage classification project. It is split into two main parts:
- `esp32/`: Contains the microcontroller code for capturing images and running the quantized model (to be developed).
- `python/`: Contains the Python code for training and exporting the garbage classification model.


All you need to do to get things up and running is:

1. Run the `preprocess.py` needs to have a zip file with all data in the root file, this will generate the numpy arrays that the model will work with 

2. Run the `train.py` this will create a model that will be saved in the esp32 folder
3. From a esp-idf terminal, `idf.py build` and then `idf.py flash monitor` the project to the board 
