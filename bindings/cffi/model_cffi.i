%module model_cffi
%{
#include <stdint.h>
#include <stddef.h>
uint32_t pygo_model_cffi_abi_version(void);
int pygo_model_cffi_validate_buffer(const void *data, size_t bytes);
%}

%include <stdint.i>
%include <stddef.i>
uint32_t pygo_model_cffi_abi_version(void);
int pygo_model_cffi_validate_buffer(const void *data, size_t bytes);
