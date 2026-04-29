#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>

bool inference_init(void);
int8_t* inference_get_input_tensor(void);
float inference_get_input_scale(void);
int inference_get_input_zero_point(void);
bool inference_run(void);

#endif // INFERENCE_H