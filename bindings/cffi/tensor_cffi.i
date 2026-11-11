%module tensor_cffi
%{
#include <stdint.h>
#include <stddef.h>
uint32_t pygo_tensor_cffi_abi_version(void);
int pygo_tensor_cffi_validate_buffer(const void *data, size_t bytes);
%}

%include <stdint.i>
%include <stddef.i>
uint32_t pygo_tensor_cffi_abi_version(void);
int pygo_tensor_cffi_validate_buffer(const void *data, size_t bytes);
