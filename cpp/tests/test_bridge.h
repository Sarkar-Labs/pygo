#ifndef PYGO_TEST_BRIDGE_H
#define PYGO_TEST_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t pygo_test_bridge_abi_version(void);
int pygo_test_bridge_self_test(void);

#ifdef __cplusplus
}
#endif

#endif
