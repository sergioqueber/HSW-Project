#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>

bool inference_init(void);
int8_t* inference_get_input_tensor(void);
bool inference_run(float* confidences);

#endif // INFERENCE_H