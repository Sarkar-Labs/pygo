#ifndef PYGO_TEST_AVX_H
#define PYGO_TEST_AVX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t pygo_test_avx_abi_version(void);
int pygo_test_avx_self_test(void);

#ifdef __cplusplus
}
#endif

#endif
