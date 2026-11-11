#ifndef PYGO_TENSOR_CFFI_H
#define PYGO_TENSOR_CFFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t pygo_tensor_cffi_abi_version(void);
int pygo_tensor_cffi_self_test(void);

#ifdef __cplusplus
}
#endif

#endif
