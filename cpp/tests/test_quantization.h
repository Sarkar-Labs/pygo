#ifndef PYGO_TEST_QUANTIZATION_H
#define PYGO_TEST_QUANTIZATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t pygo_test_quantization_abi_version(void);
int pygo_test_quantization_self_test(void);

#ifdef __cplusplus
}
#endif

#endif
