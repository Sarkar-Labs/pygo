#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct pygo_tensor_handle pygo_tensor_handle;

// Called exactly once when the last tensor handle referencing an external
// buffer is released. The callback owns responsibility for releasing its
// context/buffer.
typedef void (*pygo_external_buffer_release_fn)(void* context);

uint32_t pygo_abi_version(void);
const char* pygo_last_error(void);
void pygo_clear_error(void);

pygo_tensor_handle* pygo_tensor_zeros(const int64_t* dims, size_t ndim);
pygo_tensor_handle* pygo_tensor_ones(const int64_t* dims, size_t ndim);
pygo_tensor_handle* pygo_tensor_from_f32(const float* data, size_t count, const int64_t* dims, size_t ndim);
// Creates a tensor view over an existing float32 buffer without copying.
// The caller must provide a release callback if the buffer has managed
// ownership. The callback runs when the final tensor handle is released.
pygo_tensor_handle* pygo_tensor_from_f32_zero_copy(float* data, size_t count, const int64_t* dims, size_t ndim, void* owner_context, pygo_external_buffer_release_fn release_fn);
pygo_tensor_handle* pygo_tensor_clone(const pygo_tensor_handle* t);
void pygo_tensor_retain(pygo_tensor_handle* t);
void pygo_tensor_release(pygo_tensor_handle* t);
const float* pygo_tensor_data_f32(const pygo_tensor_handle* t);
float* pygo_tensor_mutable_data_f32(pygo_tensor_handle* t);
size_t pygo_tensor_numel(const pygo_tensor_handle* t);
size_t pygo_tensor_ndim(const pygo_tensor_handle* t);
int64_t pygo_tensor_dim(const pygo_tensor_handle* t, size_t dim);

pygo_tensor_handle* pygo_tensor_add(const pygo_tensor_handle* a, const pygo_tensor_handle* b);
pygo_tensor_handle* pygo_tensor_mul(const pygo_tensor_handle* a, const pygo_tensor_handle* b);
pygo_tensor_handle* pygo_tensor_matmul(const pygo_tensor_handle* a, const pygo_tensor_handle* b);
pygo_tensor_handle* pygo_tensor_relu(const pygo_tensor_handle* a);
pygo_tensor_handle* pygo_tensor_sum(const pygo_tensor_handle* a);

#ifdef __cplusplus
}
#endif
