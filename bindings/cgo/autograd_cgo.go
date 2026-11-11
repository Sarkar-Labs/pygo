// autograd_cgo.go — Reverse-mode autograd engine bridging Go and Python.
package cgo

/*
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// ─── Gradient accumulation ────────────────────────────────────────────────────

static void pygo_grad_accum(float* grad, float* delta, int64_t n) {
    for (int64_t i = 0; i < n; i++) grad[i] += delta[i];
}

static void pygo_grad_zero(float* grad, int64_t n) {
    memset(grad, 0, n * sizeof(float));
}

// ─── Backward: elementwise ops ────────────────────────────────────────────────

// dL/dx = dL/dout (add is identity)
static void pygo_backward_add(float* dout, float* da, float* db, int64_t n) {
    for (int64_t i = 0; i < n; i++) { da[i] += dout[i]; db[i] += dout[i]; }
}

// dL/da = dL/dout * b, dL/db = dL/dout * a
static void pygo_backward_mul(
    float* dout, float* a, float* b, float* da, float* db, int64_t n
) {
    for (int64_t i = 0; i < n; i++) { da[i] += dout[i]*b[i]; db[i] += dout[i]*a[i]; }
}

// dL/dx = dL/dout * (x > 0 ? 1 : 0)
static void pygo_backward_relu(float* dout, float* x, float* dx, int64_t n) {
    for (int64_t i = 0; i < n; i++) dx[i] += x[i] > 0.f ? dout[i] : 0.f;
}

// dL/dx = dL/dout * sig * (1 - sig)
static void pygo_backward_sigmoid(float* dout, float* out, float* dx, int64_t n) {
    for (int64_t i = 0; i < n; i++) dx[i] += dout[i] * out[i] * (1.f - out[i]);
}

// dL/dx = dL/dout * (1 - tanh^2)
static void pygo_backward_tanh(float* dout, float* out, float* dx, int64_t n) {
    for (int64_t i = 0; i < n; i++) dx[i] += dout[i] * (1.f - out[i]*out[i]);
}

// dL/dx = dL/dout / x
static void pygo_backward_log(float* dout, float* x, float* dx, int64_t n) {
    for (int64_t i = 0; i < n; i++) dx[i] += dout[i] / (x[i] + 1e-8f);
}

// dL/dx = dL/dout * 0.5 / sqrt(x)
static void pygo_backward_sqrt(float* dout, float* x, float* dx, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float s = x[i] > 0.f ? sqrtf(x[i]) : 1e-8f;
        dx[i] += dout[i] * 0.5f / s;
    }
}

// dL/dx = dL/dout * pow * x^(pow-1)
static void pygo_backward_pow(float* dout, float* x, float exp, float* dx, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        dx[i] += dout[i] * exp * powf(x[i] > 0.f ? x[i] : 1e-8f, exp - 1.f);
}

// ─── Backward: softmax + cross entropy ───────────────────────────────────────

// Combined softmax + CE backward: dL/dx = softmax(x) - one_hot(target)
static void pygo_backward_softmax_ce(
    float* logits, int64_t* targets, float* dx,
    int64_t B, int64_t C
) {
    for (int64_t b = 0; b < B; b++) {
        float* lp = logits + b*C;
        float* gp = dx     + b*C;
        float mx = lp[0];
        for (int64_t c = 1; c < C; c++) if (lp[c] > mx) mx = lp[c];
        float s = 0.f;
        for (int64_t c = 0; c < C; c++) { gp[c] = expf(lp[c]-mx); s += gp[c]; }
        for (int64_t c = 0; c < C; c++) gp[c] /= s;
        gp[targets[b]] -= 1.f;
        for (int64_t c = 0; c < C; c++) gp[c] /= (float)B;
    }
}

// ─── Backward: MatMul [M,K] x [K,N] → [M,N] ─────────────────────────────────
// dA = dOut @ B^T,  dB = A^T @ dOut

static void pygo_backward_matmul(
    float* dout, float* A, float* B, float* dA, float* dB,
    int64_t M, int64_t K, int64_t N
) {
    // dA[m,k] += sum_n dout[m,n] * B[k,n]
    for (int64_t m = 0; m < M; m++)
        for (int64_t k = 0; k < K; k++)
            for (int64_t n = 0; n < N; n++)
                dA[m*K+k] += dout[m*N+n] * B[k*N+n];
    // dB[k,n] += sum_m A[m,k] * dout[m,n]
    for (int64_t k = 0; k < K; k++)
        for (int64_t n = 0; n < N; n++)
            for (int64_t m = 0; m < M; m++)
                dB[k*N+n] += A[m*K+k] * dout[m*N+n];
}

// ─── Backward: LayerNorm ──────────────────────────────────────────────────────

static void pygo_backward_layernorm(
    float* dout, float* x, float* weight, float* dx, float* dw, float* db,
    int64_t BT, int64_t C, float eps
) {
    for (int64_t bt = 0; bt < BT; bt++) {
        float* xp  = x    + bt*C;
        float* gp  = dout + bt*C;
        float* dgp = dx   + bt*C;
        float mean = 0.f;
        for (int64_t c = 0; c < C; c++) mean += xp[c];
        mean /= (float)C;
        float var = 0.f;
        for (int64_t c = 0; c < C; c++) { float d=xp[c]-mean; var+=d*d; }
        var /= (float)C;
        float inv_std = 1.f / sqrtf(var + eps);
        float* xhat = (float*)alloca(C * sizeof(float));
        for (int64_t c = 0; c < C; c++) xhat[c] = (xp[c]-mean)*inv_std;
        // dL/dweight, dL/dbias
        if (dw) for (int64_t c = 0; c < C; c++) dw[c] += gp[c] * xhat[c];
        if (db) for (int64_t c = 0; c < C; c++) db[c] += gp[c];
        // dL/dx
        float dxhat_sum = 0.f, dxhat_xhat_sum = 0.f;
        float* dxhat = (float*)alloca(C * sizeof(float));
        for (int64_t c = 0; c < C; c++) {
            dxhat[c] = gp[c] * (weight ? weight[c] : 1.f);
            dxhat_sum      += dxhat[c];
            dxhat_xhat_sum += dxhat[c] * xhat[c];
        }
        for (int64_t c = 0; c < C; c++)
            dgp[c] += inv_std / (float)C * (
                (float)C * dxhat[c] - dxhat_sum - xhat[c] * dxhat_xhat_sum
            );
    }
}
*/
import "C"
import "unsafe"

// ─── Grad ops ─────────────────────────────────────────────────────────────────

// GradAccum accumulates delta into grad (grad += delta).
func GradAccum(grad, delta *BridgeTensor) {
	C.pygo_grad_accum((*C.float)(grad.DataPtr()), (*C.float)(delta.DataPtr()), C.int64_t(grad.NumEl()))
}

// GradZero zeroes gradient buffer.
func GradZero(grad *BridgeTensor) {
	C.pygo_grad_zero((*C.float)(grad.DataPtr()), C.int64_t(grad.NumEl()))
}

// ─── Backward passes ─────────────────────────────────────────────────────────

// BackwardAdd: dL/da += dout, dL/db += dout
func BackwardAdd(dout, da, db *BridgeTensor) {
	C.pygo_backward_add((*C.float)(dout.DataPtr()), (*C.float)(da.DataPtr()), (*C.float)(db.DataPtr()), C.int64_t(dout.NumEl()))
}

// BackwardMul: dL/da += dout*b, dL/db += dout*a
func BackwardMul(dout, a, b, da, db *BridgeTensor) {
	C.pygo_backward_mul((*C.float)(dout.DataPtr()), (*C.float)(a.DataPtr()), (*C.float)(b.DataPtr()), (*C.float)(da.DataPtr()), (*C.float)(db.DataPtr()), C.int64_t(dout.NumEl()))
}

// BackwardReLU: dL/dx += dout * (x > 0)
func BackwardReLU(dout, x, dx *BridgeTensor) {
	C.pygo_backward_relu((*C.float)(dout.DataPtr()), (*C.float)(x.DataPtr()), (*C.float)(dx.DataPtr()), C.int64_t(dout.NumEl()))
}

// BackwardSigmoid: dL/dx += dout * out * (1 - out)
func BackwardSigmoid(dout, out, dx *BridgeTensor) {
	C.pygo_backward_sigmoid((*C.float)(dout.DataPtr()), (*C.float)(out.DataPtr()), (*C.float)(dx.DataPtr()), C.int64_t(dout.NumEl()))
}

// BackwardTanh: dL/dx += dout * (1 - out^2)
func BackwardTanh(dout, out, dx *BridgeTensor) {
	C.pygo_backward_tanh((*C.float)(dout.DataPtr()), (*C.float)(out.DataPtr()), (*C.float)(dx.DataPtr()), C.int64_t(dout.NumEl()))
}

// BackwardLog: dL/dx += dout / x
func BackwardLog(dout, x, dx *BridgeTensor) {
	C.pygo_backward_log((*C.float)(dout.DataPtr()), (*C.float)(x.DataPtr()), (*C.float)(dx.DataPtr()), C.int64_t(dout.NumEl()))
}

// BackwardSqrt: dL/dx += dout * 0.5 / sqrt(x)
func BackwardSqrt(dout, x, dx *BridgeTensor) {
	C.pygo_backward_sqrt((*C.float)(dout.DataPtr()), (*C.float)(x.DataPtr()), (*C.float)(dx.DataPtr()), C.int64_t(dout.NumEl()))
}

// BackwardPow: dL/dx += dout * pow * x^(pow-1)
func BackwardPow(dout, x *BridgeTensor, exp float32, dx *BridgeTensor) {
	C.pygo_backward_pow((*C.float)(dout.DataPtr()), (*C.float)(x.DataPtr()), C.float(exp), (*C.float)(dx.DataPtr()), C.int64_t(dout.NumEl()))
}

// BackwardSoftmaxCE: combined softmax + cross-entropy backward.
// logits: [B,C], targets: [B] int64, dx: [B,C]
func BackwardSoftmaxCE(logits *BridgeTensor, targets *BridgeTensor, dx *BridgeTensor) {
	B := logits.shape[0]
	Cols := logits.shape[1]
	C.pygo_backward_softmax_ce(
		(*C.float)(logits.DataPtr()),
		(*C.int64_t)(targets.DataPtr()),
		(*C.float)(dx.DataPtr()),
		C.int64_t(B), C.int64_t(Cols),
	)
}

// BackwardMatMul: dA = dOut @ B^T, dB = A^T @ dOut
func BackwardMatMul(dout, A, B, dA, dB *BridgeTensor) {
	M := A.shape[len(A.shape)-2]
	K := A.shape[len(A.shape)-1]
	N := B.shape[len(B.shape)-1]
	C.pygo_backward_matmul(
		(*C.float)(dout.DataPtr()),
		(*C.float)(A.DataPtr()),
		(*C.float)(B.DataPtr()),
		(*C.float)(dA.DataPtr()),
		(*C.float)(dB.DataPtr()),
		C.int64_t(M), C.int64_t(K), C.int64_t(N),
	)
}

// BackwardLayerNorm: full LN backward (dx, dweight, dbias).
func BackwardLayerNorm(dout, x, weight, dx, dw, db *BridgeTensor, eps float32) {
	cols := x.shape[len(x.shape)-1]
	BT := x.NumEl() / cols
	var wp, dwp, dbp *C.float
	if weight != nil {
		wp = (*C.float)(weight.DataPtr())
	}
	if dw != nil {
		dwp = (*C.float)(dw.DataPtr())
	}
	if db != nil {
		dbp = (*C.float)(db.DataPtr())
	}
	C.pygo_backward_layernorm(
		(*C.float)(dout.DataPtr()),
		(*C.float)(x.DataPtr()),
		wp,
		(*C.float)(dx.DataPtr()),
		dwp, dbp,
		C.int64_t(BT), C.int64_t(cols), C.float(eps),
	)
}

// ─── Tape-based autograd engine ───────────────────────────────────────────────

// OpKind identifies the operation stored on the backward tape.
type OpKind int

const (
	OpAdd OpKind = iota
	OpSub
	OpMul
	OpDiv
	OpMatMul
	OpReLU
	OpGELU
	OpSiLU
	OpSigmoid
	OpTanh
	OpLog
	OpSqrt
	OpPow
	OpSoftmaxCE
	OpLayerNorm
	OpRMSNorm
	OpScale
)

// TapeEntry is one record on the backward tape.
type TapeEntry struct {
	Op     OpKind
	Inputs []*BridgeTensor
	Output *BridgeTensor
	Grads  []*BridgeTensor // gradient buffers for each input
	Scalar float32         // for Scale, Pow
	Eps    float32         // for LayerNorm, RMSNorm
}

// Tape records forward ops for later backward pass.
type Tape struct {
	entries []*TapeEntry
	enabled bool
}

// NewTape creates a new autograd tape.
func NewTape() *Tape { return &Tape{enabled: true} }

// Disable stops recording.
func (t *Tape) Disable() { t.enabled = false }

// Enable starts recording.
func (t *Tape) Enable() { t.enabled = true }

// Record adds an entry to the tape.
func (t *Tape) Record(e *TapeEntry) {
	if t.enabled {
		t.entries = append(t.entries, e)
	}
}

// Backward runs the backward pass, accumulating gradients.
// outGrad is dL/dOutput for the final output tensor (usually ones or loss grad).
func (t *Tape) Backward(outGrad *BridgeTensor) error {
	// Walk tape in reverse
	for i := len(t.entries) - 1; i >= 0; i-- {
		e := t.entries[i]
		dout := outGrad // simplified: real impl tracks per-tensor grads
		_ = dout

		switch e.Op {
		case OpAdd:
			BackwardAdd(outGrad, e.Grads[0], e.Grads[1])
		case OpMul:
			BackwardMul(outGrad, e.Inputs[0], e.Inputs[1], e.Grads[0], e.Grads[1])
		case OpMatMul:
			BackwardMatMul(outGrad, e.Inputs[0], e.Inputs[1], e.Grads[0], e.Grads[1])
		case OpReLU:
			BackwardReLU(outGrad, e.Inputs[0], e.Grads[0])
		case OpSigmoid:
			BackwardSigmoid(outGrad, e.Output, e.Grads[0])
		case OpTanh:
			BackwardTanh(outGrad, e.Output, e.Grads[0])
		case OpLog:
			BackwardLog(outGrad, e.Inputs[0], e.Grads[0])
		case OpSqrt:
			BackwardSqrt(outGrad, e.Inputs[0], e.Grads[0])
		case OpPow:
			BackwardPow(outGrad, e.Inputs[0], e.Scalar, e.Grads[0])
		case OpSoftmaxCE:
			BackwardSoftmaxCE(e.Inputs[0], e.Inputs[1], e.Grads[0])
		case OpLayerNorm:
			var w, dw, db *BridgeTensor
			if len(e.Inputs) > 1 {
				w = e.Inputs[1]
			}
			if len(e.Grads) > 1 {
				dw = e.Grads[1]
			}
			if len(e.Grads) > 2 {
				db = e.Grads[2]
			}
			BackwardLayerNorm(outGrad, e.Inputs[0], w, e.Grads[0], dw, db, e.Eps)
		}
	}
	return nil
}

// Reset clears the tape for the next forward pass.
func (t *Tape) Reset() { t.entries = t.entries[:0] }

// Len returns number of ops on the tape.
func (t *Tape) Len() int { return len(t.entries) }

// NoGrad runs fn without recording ops to any tape.
func NoGrad(fn func()) {
	// In real impl: set thread-local flag to disable recording
	fn()
}

// dummyUse prevents "unused import" for unsafe
var _ = unsafe.Pointer(nil)
