#ifndef PYGO_TEST_METAL_H
#define PYGO_TEST_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t pygo_test_metal_abi_version(void);
int pygo_test_metal_self_test(void);

#ifdef __cplusplus
}
#endif

#endif
