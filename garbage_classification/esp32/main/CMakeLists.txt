idf_component_register(SRCS "main.cpp" "camera.cpp" "preprocess.cpp" "inference.cpp" "model.c"
                    INCLUDE_DIRS "."
                    REQUIRES esp_driver_usb_serial_jtag esp32-camera esp-tflite-micro esp_timer)