#ifndef PYGO_TEST_KERNELS_H
#define PYGO_TEST_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t pygo_test_kernels_abi_version(void);
int pygo_test_kernels_self_test(void);

#ifdef __cplusplus
}
#endif

#endif
