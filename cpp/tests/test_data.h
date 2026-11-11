#ifndef PYGO_TEST_DATA_H
#define PYGO_TEST_DATA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t pygo_test_data_abi_version(void);
int pygo_test_data_self_test(void);

#ifdef __cplusplus
}
#endif

#endif
