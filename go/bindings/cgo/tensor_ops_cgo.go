// tensor_ops_cgo.go — CGo tensor operations: matmul, elementwise, reduce.
package cgo

/*
#cgo pkg-config: python3-embed
#cgo LDFLAGS: -lm -ldl
#cgo CFLAGS: -I../../cpp/tensor -O3


#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ─── Elementwise ops (CPU fallback) ──────────────────────────────────────────

static void pygo_add_f32(float* a, float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}
static void pygo_sub_f32(float* a, float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] - b[i];
}
static void pygo_mul_f32(float* a, float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] * b[i];
}
static void pygo_div_f32(float* a, float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = b[i] != 0.f ? a[i]/b[i] : 0.f;
}
static void pygo_scale_f32(float* a, float scalar, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] * scalar;
}
static void pygo_add_scalar_f32(float* a, float scalar, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] + scalar;
}
static void pygo_relu_f32(float* a, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] > 0.f ? a[i] : 0.f;
}
static void pygo_gelu_f32(float* a, float* out, int64_t n) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    for (int64_t i = 0; i < n; i++) {
        float x = a[i];
        float inner = c * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.f + tanhf(inner));
    }
}
static void pygo_silu_f32(float* a, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = a[i] / (1.f + expf(-a[i]));
}
static void pygo_tanh_f32(float* a, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = tanhf(a[i]);
}
static void pygo_sigmoid_f32(float* a, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = 1.f / (1.f + expf(-a[i]));
}
static void pygo_exp_f32(float* a, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = expf(a[i]);
}
static void pygo_log_f32(float* a, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = logf(a[i] > 0.f ? a[i] : 1e-8f);
}
static void pygo_sqrt_f32(float* a, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = sqrtf(a[i] > 0.f ? a[i] : 0.f);
}
static void pygo_pow_f32(float* a, float exp, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = powf(a[i], exp);
}
static void pygo_abs_f32(float* a, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = fabsf(a[i]);
}
static void pygo_neg_f32(float* a, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) out[i] = -a[i];
}
static void pygo_clamp_f32(float* a, float mn, float mx, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float v = a[i];
        out[i] = v < mn ? mn : (v > mx ? mx : v);
    }
}

// ─── Reductions ───────────────────────────────────────────────────────────────

static float pygo_sum_f32(float* a, int64_t n) {
    float s = 0.f;
    for (int64_t i = 0; i < n; i++) s += a[i];
    return s;
}
static float pygo_mean_f32(float* a, int64_t n) {
    return n > 0 ? pygo_sum_f32(a, n) / (float)n : 0.f;
}
static float pygo_max_f32(float* a, int64_t n) {
    float m = a[0];
    for (int64_t i = 1; i < n; i++) if (a[i] > m) m = a[i];
    return m;
}
static float pygo_min_f32(float* a, int64_t n) {
    float m = a[0];
    for (int64_t i = 1; i < n; i++) if (a[i] < m) m = a[i];
    return m;
}
static float pygo_norm_l2_f32(float* a, int64_t n) {
    float s = 0.f;
    for (int64_t i = 0; i < n; i++) s += a[i] * a[i];
    return sqrtf(s);
}

// ─── Softmax (numerically stable) ────────────────────────────────────────────

static void pygo_softmax_f32(float* a, float* out, int64_t n) {
    float mx = pygo_max_f32(a, n);
    float s = 0.f;
    for (int64_t i = 0; i < n; i++) { out[i] = expf(a[i] - mx); s += out[i]; }
    if (s > 0.f) for (int64_t i = 0; i < n; i++) out[i] /= s;
}

// ─── Naive MatMul (CPU reference) [M,K] x [K,N] -> [M,N] ────────────────────

static void pygo_matmul_f32(
    float* A, float* B, float* C,
    int64_t M, int64_t K, int64_t N
) {
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.f;
            for (int64_t k = 0; k < K; k++)
                acc += A[m*K+k] * B[k*N+n];
            C[m*N+n] = acc;
        }
    }
}

// ─── Layer Norm ───────────────────────────────────────────────────────────────

static void pygo_layer_norm_f32(
    float* x, float* weight, float* bias, float* out,
    int64_t B, int64_t T, int64_t C, float eps
) {
    for (int64_t bt = 0; bt < B*T; bt++) {
        float* xp = x   + bt*C;
        float* op = out + bt*C;
        float mean = 0.f;
        for (int64_t c = 0; c < C; c++) mean += xp[c];
        mean /= (float)C;
        float var = 0.f;
        for (int64_t c = 0; c < C; c++) { float d = xp[c]-mean; var += d*d; }
        var /= (float)C;
        float inv_std = 1.f / sqrtf(var + eps);
        for (int64_t c = 0; c < C; c++)
            op[c] = (xp[c]-mean)*inv_std * (weight?weight[c]:1.f) + (bias?bias[c]:0.f);
    }
}

// ─── RMSNorm ─────────────────────────────────────────────────────────────────

static void pygo_rms_norm_f32(
    float* x, float* weight, float* out,
    int64_t rows, int64_t cols, float eps
) {
    for (int64_t r = 0; r < rows; r++) {
        float* xp = x   + r*cols;
        float* op = out + r*cols;
        float ss = 0.f;
        for (int64_t c = 0; c < cols; c++) ss += xp[c]*xp[c];
        float inv = 1.f / sqrtf(ss/(float)cols + eps);
        for (int64_t c = 0; c < cols; c++)
            op[c] = xp[c]*inv*(weight?weight[c]:1.f);
    }
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// ─── Elementwise ops ─────────────────────────────────────────────────────────

func Add(a, b, out *BridgeTensor) error {
	if err := checkSameShape(a, b, out); err != nil {
		return err
	}
	C.pygo_add_f32((*C.float)(a.DataPtr()), (*C.float)(b.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Sub(a, b, out *BridgeTensor) error {
	if err := checkSameShape(a, b, out); err != nil {
		return err
	}
	C.pygo_sub_f32((*C.float)(a.DataPtr()), (*C.float)(b.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Mul(a, b, out *BridgeTensor) error {
	if err := checkSameShape(a, b, out); err != nil {
		return err
	}
	C.pygo_mul_f32((*C.float)(a.DataPtr()), (*C.float)(b.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Div(a, b, out *BridgeTensor) error {
	if err := checkSameShape(a, b, out); err != nil {
		return err
	}
	C.pygo_div_f32((*C.float)(a.DataPtr()), (*C.float)(b.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Scale(a *BridgeTensor, scalar float32, out *BridgeTensor) error {
	C.pygo_scale_f32((*C.float)(a.DataPtr()), C.float(scalar), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func AddScalar(a *BridgeTensor, scalar float32, out *BridgeTensor) error {
	C.pygo_add_scalar_f32((*C.float)(a.DataPtr()), C.float(scalar), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func ReLU(a, out *BridgeTensor) error {
	C.pygo_relu_f32((*C.float)(a.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func GELU(a, out *BridgeTensor) error {
	C.pygo_gelu_f32((*C.float)(a.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func SiLU(a, out *BridgeTensor) error {
	C.pygo_silu_f32((*C.float)(a.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Tanh(a, out *BridgeTensor) error {
	C.pygo_tanh_f32((*C.float)(a.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Sigmoid(a, out *BridgeTensor) error {
	C.pygo_sigmoid_f32((*C.float)(a.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Exp(a, out *BridgeTensor) error {
	C.pygo_exp_f32((*C.float)(a.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Log(a, out *BridgeTensor) error {
	C.pygo_log_f32((*C.float)(a.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Sqrt(a, out *BridgeTensor) error {
	C.pygo_sqrt_f32((*C.float)(a.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Pow(a *BridgeTensor, exp float32, out *BridgeTensor) error {
	C.pygo_pow_f32((*C.float)(a.DataPtr()), C.float(exp), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Abs(a, out *BridgeTensor) error {
	C.pygo_abs_f32((*C.float)(a.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Neg(a, out *BridgeTensor) error {
	C.pygo_neg_f32((*C.float)(a.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Clamp(a *BridgeTensor, mn, mx float32, out *BridgeTensor) error {
	C.pygo_clamp_f32((*C.float)(a.DataPtr()), C.float(mn), C.float(mx), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

func Softmax(a, out *BridgeTensor) error {
	C.pygo_softmax_f32((*C.float)(a.DataPtr()), (*C.float)(out.DataPtr()), C.int64_t(a.NumEl()))
	return nil
}

// ─── Reductions ───────────────────────────────────────────────────────────────

func Sum(a *BridgeTensor) float32 {
	return float32(C.pygo_sum_f32((*C.float)(a.DataPtr()), C.int64_t(a.NumEl())))
}

func Mean(a *BridgeTensor) float32 {
	return float32(C.pygo_mean_f32((*C.float)(a.DataPtr()), C.int64_t(a.NumEl())))
}

func Max(a *BridgeTensor) float32 {
	return float32(C.pygo_max_f32((*C.float)(a.DataPtr()), C.int64_t(a.NumEl())))
}

func Min(a *BridgeTensor) float32 {
	return float32(C.pygo_min_f32((*C.float)(a.DataPtr()), C.int64_t(a.NumEl())))
}

func NormL2(a *BridgeTensor) float32 {
	return float32(C.pygo_norm_l2_f32((*C.float)(a.DataPtr()), C.int64_t(a.NumEl())))
}

// ─── MatMul [M,K] x [K,N] → [M,N] ───────────────────────────────────────────

func MatMul(a, b, out *BridgeTensor) error {
	if len(a.shape) < 2 || len(b.shape) < 2 {
		return fmt.Errorf("MatMul: inputs must be at least 2D")
	}
	M := a.shape[len(a.shape)-2]
	K := a.shape[len(a.shape)-1]
	N := b.shape[len(b.shape)-1]
	if b.shape[len(b.shape)-2] != K {
		return fmt.Errorf("MatMul: shape mismatch [%d,%d] x [%d,%d]", M, K, b.shape[len(b.shape)-2], N)
	}
	C.pygo_matmul_f32(
		(*C.float)(a.DataPtr()),
		(*C.float)(b.DataPtr()),
		(*C.float)(out.DataPtr()),
		C.int64_t(M), C.int64_t(K), C.int64_t(N),
	)
	return nil
}

// ─── LayerNorm ────────────────────────────────────────────────────────────────

func LayerNorm(x, weight, bias, out *BridgeTensor, eps float32) error {
	if len(x.shape) < 2 {
		return fmt.Errorf("LayerNorm: input must be at least 2D")
	}
	C := x.shape[len(x.shape)-1]
	BT := x.NumEl() / C
	var wp, bp *C.float
	if weight != nil {
		wp = (*C.float)(weight.DataPtr())
	}
	if bias != nil {
		bp = (*C.float)(bias.DataPtr())
	}
	C_layernorm((*C.float)(x.DataPtr()), wp, bp, (*C.float)(out.DataPtr()),
		C_int64_t(BT), C_int64_t(C), C.float(eps))
	return nil
}

func C_layernorm(x, w, b, out *C.float, BT, Cols C.int64_t, eps C.float) {
	C.pygo_layer_norm_f32(x, w, b, out, C.int64_t(1), BT, Cols, eps)
}

func C_int64_t(v int64) C.int64_t { return C.int64_t(v) }

// ─── RMSNorm ─────────────────────────────────────────────────────────────────

func RMSNorm(x, weight, out *BridgeTensor, eps float32) error {
	if len(x.shape) < 2 {
		return fmt.Errorf("RMSNorm: input must be at least 2D")
	}
	cols := x.shape[len(x.shape)-1]
	rows := x.NumEl() / cols
	var wp *C.float
	if weight != nil {
		wp = (*C.float)(weight.DataPtr())
	}
	C.pygo_rms_norm_f32((*C.float)(x.DataPtr()), wp, (*C.float)(out.DataPtr()),
		C.int64_t(rows), C.int64_t(cols), C.float(eps))
	return nil
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

func checkSameShape(tensors ...*BridgeTensor) error {
	if len(tensors) < 2 {
		return nil
	}
	ref := tensors[0].NumEl()
	for _, t := range tensors[1:] {
		if t.NumEl() != ref {
			return fmt.Errorf("shape mismatch: %v vs %v", tensors[0].shape, t.shape)
		}
	}
	return nil
}

// Fill sets all elements to value.
func Fill(t *BridgeTensor, value float32) {
	s := t.Float32Slice()
	for i := range s {
		s[i] = value
	}
}

// Zeros fills tensor with 0.
func Zeros(t *BridgeTensor) { Fill(t, 0) }

// Ones fills tensor with 1.
func Ones(t *BridgeTensor) { Fill(t, 1) }

// Item returns the scalar value of a 1-element tensor.
func Item(t *BridgeTensor) (float32, error) {
	if t.NumEl() != 1 {
		return 0, fmt.Errorf("Item: tensor has %d elements, expected 1", t.NumEl())
	}
	return t.Float32Slice()[0], nil
}

// CopyTo copies data from src to dst (must have same NumEl).
func CopyTo(src, dst *BridgeTensor) error {
	if src.NumEl() != dst.NumEl() {
		return fmt.Errorf("CopyTo: size mismatch %d vs %d", src.NumEl(), dst.NumEl())
	}
	n := src.NumEl() * int64(src.dtype.ItemSize())
	C.memcpy(dst.DataPtr(), src.DataPtr(), C.size_t(n))
	return nil
}

// DataPtrOffset returns a pointer offset into tensor data (for slicing).
func DataPtrOffset(t *BridgeTensor, offset int64) unsafe.Pointer {
	base := uintptr(t.DataPtr())
	return unsafe.Pointer(base + uintptr(offset)*uintptr(t.dtype.ItemSize()))
}
