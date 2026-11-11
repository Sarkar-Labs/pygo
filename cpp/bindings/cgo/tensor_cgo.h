#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pygo_tensor_handle* PyGoTensorHandle;
typedef enum PyGoOpCode {
    PyGoOp_Add = 0, PyGoOp_Sub, PyGoOp_Mul, PyGoOp_Div, PyGoOp_MatMul,
    PyGoOp_Sqrt, PyGoOp_Abs, PyGoOp_Exp, PyGoOp_Log, PyGoOp_Sin,
    PyGoOp_Cos, PyGoOp_Tanh, PyGoOp_Neg, PyGoOp_ReLU, PyGoOp_GELU,
    PyGoOp_SiLU, PyGoOp_Sigmoid
} PyGoOpCode;

uint32_t pygo_tensor_abi_version(void);
PyGoTensorHandle PyGoTensor_Zeros(const int64_t*, int, int, int, int);
PyGoTensorHandle PyGoTensor_Ones(const int64_t*, int, int, int, int);
PyGoTensorHandle PyGoTensor_Randn(const int64_t*, int, int, int, int);
PyGoTensorHandle PyGoTensor_Arange(double, double, double, int, int, int);
PyGoTensorHandle PyGoTensor_FromData(const void*, size_t, const int64_t*, int, int, int, int);
void PyGoTensor_Release(PyGoTensorHandle);
void* PyGoTensor_DataF32(PyGoTensorHandle);
PyGoTensorHandle PyGoTensor_BinOp(PyGoTensorHandle, PyGoTensorHandle, PyGoOpCode);
PyGoTensorHandle PyGoTensor_UnaryOp(PyGoTensorHandle, PyGoOpCode);
PyGoTensorHandle PyGoTensor_AddScalar(PyGoTensorHandle, double);
PyGoTensorHandle PyGoTensor_MulScalar(PyGoTensorHandle, double);
PyGoTensorHandle PyGoTensor_Pow(PyGoTensorHandle, double);
PyGoTensorHandle PyGoTensor_Softmax(PyGoTensorHandle, int);
PyGoTensorHandle PyGoTensor_ReduceSum(PyGoTensorHandle, int, bool);
PyGoTensorHandle PyGoTensor_ReduceMean(PyGoTensorHandle, int, bool);
PyGoTensorHandle PyGoTensor_ReduceMax(PyGoTensorHandle, int, bool);
PyGoTensorHandle PyGoTensor_Argmax(PyGoTensorHandle, int);
double PyGoTensor_Item(PyGoTensorHandle);
PyGoTensorHandle PyGoTensor_Reshape(PyGoTensorHandle, const int64_t*, int);
PyGoTensorHandle PyGoTensor_Flatten(PyGoTensorHandle, int, int);
PyGoTensorHandle PyGoTensor_Transpose(PyGoTensorHandle, int, int);
PyGoTensorHandle PyGoTensor_Permute(PyGoTensorHandle, const int*, int);
PyGoTensorHandle PyGoTensor_Squeeze(PyGoTensorHandle, int);
PyGoTensorHandle PyGoTensor_Unsqueeze(PyGoTensorHandle, int);
PyGoTensorHandle PyGoTensor_To(PyGoTensorHandle, int, int);
PyGoTensorHandle PyGoTensor_Cast(PyGoTensorHandle, int);
void PyGoTensor_Backward(PyGoTensorHandle, PyGoTensorHandle, bool);
void PyGoTensor_SetRequiresGrad(PyGoTensorHandle, bool);
PyGoTensorHandle PyGoTensor_GetGrad(PyGoTensorHandle);
void PyGoTensor_ZeroGrad(PyGoTensorHandle);
PyGoTensorHandle PyGoTensor_Detach(PyGoTensorHandle);
PyGoTensorHandle PyGoTensor_Clone(PyGoTensorHandle);
PyGoTensorHandle PyGoTensor_Contiguous(PyGoTensorHandle);
char* PyGoTensor_ToString(PyGoTensorHandle, int);
int PyGoTensor_NDim(PyGoTensorHandle);
int64_t PyGoTensor_Shape(PyGoTensorHandle, int64_t* out);

#ifdef __cplusplus
}
#endif
