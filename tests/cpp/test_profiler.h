#ifndef PYGO_TEST_PROFILER_H
#define PYGO_TEST_PROFILER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t pygo_test_profiler_abi_version(void);
int pygo_test_profiler_self_test(void);

#ifdef __cplusplus
}
#endif

#endif
