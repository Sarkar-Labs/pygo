// optim_cgo.go — Fused optimizer kernels via CGo (AdamW, Lion, SGD).
package cgo

/*
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// ─── Fused AdamW step ─────────────────────────────────────────────────────────
// Performs: m = b1*m + (1-b1)*g
//           v = b2*v + (1-b2)*g^2
//           m_hat = m / (1 - b1^t)
//           v_hat = v / (1 - b2^t)
//           p = p * (1 - lr*wd) - lr * m_hat / (sqrt(v_hat) + eps)

static void pygo_adamw_step_f32(
    float* param, float* grad, float* m, float* v,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2,
    int64_t n
) {
    for (int64_t i = 0; i < n; i++) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.f - beta1) * g;
        v[i] = beta2 * v[i] + (1.f - beta2) * g * g;
        float mh = m[i] / bc1;
        float vh = v[i] / bc2;
        param[i] *= (1.f - lr * wd);
        param[i] -= lr * mh / (sqrtf(vh) + eps);
    }
}

// ─── Fused SGD with momentum ──────────────────────────────────────────────────

static void pygo_sgd_step_f32(
    float* param, float* grad, float* vel,
    float lr, float momentum, float wd, float dampening,
    int nesterov, int64_t n
) {
    for (int64_t i = 0; i < n; i++) {
        float g = grad[i] + wd * param[i];
        if (vel) {
            vel[i] = momentum * vel[i] + (1.f - dampening) * g;
            g = nesterov ? g + momentum * vel[i] : vel[i];
        }
        param[i] -= lr * g;
    }
}

// ─── Lion optimizer step ──────────────────────────────────────────────────────
// update = sign(b1*m + (1-b1)*g)
// p = p * (1 - lr*wd) - lr * update
// m = b2*m + (1-b2)*g

static void pygo_lion_step_f32(
    float* param, float* grad, float* m,
    float lr, float beta1, float beta2, float wd,
    int64_t n
) {
    for (int64_t i = 0; i < n; i++) {
        float interp = beta1 * m[i] + (1.f - beta1) * grad[i];
        float update = interp > 0.f ? 1.f : (interp < 0.f ? -1.f : 0.f);
        param[i] *= (1.f - lr * wd);
        param[i] -= lr * update;
        m[i] = beta2 * m[i] + (1.f - beta2) * grad[i];
    }
}

// ─── LAMB step ────────────────────────────────────────────────────────────────

static void pygo_lamb_step_f32(
    float* param, float* grad, float* m, float* v,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, float trust_clip,
    int64_t n
) {
    // Compute adam update
    float adam_norm = 0.f, w_norm = 0.f;
    float* adam = (float*)alloca(n * sizeof(float));
    for (int64_t i = 0; i < n; i++) {
        m[i] = beta1 * m[i] + (1.f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.f - beta2) * grad[i] * grad[i];
        adam[i] = (m[i]/bc1) / (sqrtf(v[i]/bc2) + eps) + wd * param[i];
        adam_norm += adam[i] * adam[i];
        w_norm    += param[i] * param[i];
    }
    adam_norm = sqrtf(adam_norm);
    w_norm    = sqrtf(w_norm);
    float trust = (w_norm > 0.f && adam_norm > 0.f)
                ? w_norm / adam_norm : 1.f;
    if (trust > trust_clip) trust = trust_clip;
    for (int64_t i = 0; i < n; i++)
        param[i] -= lr * trust * adam[i];
}

// ─── Gradient clipping (in-place) ─────────────────────────────────────────────

static float pygo_grad_norm_f32(float** grads, int64_t* sizes, int num_params) {
    float total = 0.f;
    for (int p = 0; p < num_params; p++) {
        for (int64_t i = 0; i < sizes[p]; i++)
            total += grads[p][i] * grads[p][i];
    }
    return sqrtf(total);
}

static void pygo_clip_grad_norm_f32(
    float** grads, int64_t* sizes, int num_params,
    float max_norm
) {
    float norm = pygo_grad_norm_f32(grads, sizes, num_params);
    if (norm > max_norm) {
        float coef = max_norm / (norm + 1e-6f);
        for (int p = 0; p < num_params; p++)
            for (int64_t i = 0; i < sizes[p]; i++)
                grads[p][i] *= coef;
    }
}

// ─── EMA (exponential moving average of weights) ──────────────────────────────

static void pygo_ema_update_f32(
    float* shadow, float* param, float decay, int64_t n
) {
    for (int64_t i = 0; i < n; i++)
        shadow[i] = decay * shadow[i] + (1.f - decay) * param[i];
}
*/
import "C"
import (
	"fmt"
	"math"
	"unsafe"
)

// ─── AdamW ───────────────────────────────────────────────────────────────────

// AdamWState holds first and second moment buffers for one parameter.
type AdamWState struct {
	M, V *BridgeTensor
	Step int64
}

// NewAdamWState allocates moment buffers matching param shape.
func NewAdamWState(param *BridgeTensor) (*AdamWState, error) {
	m, err := NewBridgeTensor(param.shape, Float32, param.device)
	if err != nil {
		return nil, err
	}
	v, err := NewBridgeTensor(param.shape, Float32, param.device)
	if err != nil {
		return nil, err
	}
	Zeros(m)
	Zeros(v)
	return &AdamWState{M: m, V: v}, nil
}

// AdamWStep performs one fused AdamW parameter update.
func AdamWStep(
	param, grad *BridgeTensor,
	state *AdamWState,
	lr, beta1, beta2, eps, wd float32,
) error {
	state.Step++
	bc1 := float32(1.0 - math.Pow(float64(beta1), float64(state.Step)))
	bc2 := float32(1.0 - math.Pow(float64(beta2), float64(state.Step)))
	C.pygo_adamw_step_f32(
		(*C.float)(param.DataPtr()),
		(*C.float)(grad.DataPtr()),
		(*C.float)(state.M.DataPtr()),
		(*C.float)(state.V.DataPtr()),
		C.float(lr), C.float(beta1), C.float(beta2),
		C.float(eps), C.float(wd),
		C.float(bc1), C.float(bc2),
		C.int64_t(param.NumEl()),
	)
	return nil
}

// ─── SGD ─────────────────────────────────────────────────────────────────────

// SGDState holds velocity buffer for one parameter.
type SGDState struct {
	Velocity *BridgeTensor
}

// NewSGDState allocates velocity buffer.
func NewSGDState(param *BridgeTensor) (*SGDState, error) {
	vel, err := NewBridgeTensor(param.shape, Float32, param.device)
	if err != nil {
		return nil, err
	}
	Zeros(vel)
	return &SGDState{Velocity: vel}, nil
}

// SGDStep performs one SGD step with optional momentum.
func SGDStep(
	param, grad *BridgeTensor,
	state *SGDState,
	lr, momentum, wd, dampening float32,
	nesterov bool,
) error {
	var velPtr *C.float
	if state != nil && state.Velocity != nil {
		velPtr = (*C.float)(state.Velocity.DataPtr())
	}
	nesterovInt := 0
	if nesterov {
		nesterovInt = 1
	}
	C.pygo_sgd_step_f32(
		(*C.float)(param.DataPtr()),
		(*C.float)(grad.DataPtr()),
		velPtr,
		C.float(lr), C.float(momentum), C.float(wd), C.float(dampening),
		C.int(nesterovInt),
		C.int64_t(param.NumEl()),
	)
	return nil
}

// ─── Lion ────────────────────────────────────────────────────────────────────

// LionState holds momentum buffer for one parameter.
type LionState struct {
	M *BridgeTensor
}

// NewLionState allocates momentum buffer.
func NewLionState(param *BridgeTensor) (*LionState, error) {
	m, err := NewBridgeTensor(param.shape, Float32, param.device)
	if err != nil {
		return nil, err
	}
	Zeros(m)
	return &LionState{M: m}, nil
}

// LionStep performs one Lion optimizer step.
func LionStep(
	param, grad *BridgeTensor,
	state *LionState,
	lr, beta1, beta2, wd float32,
) error {
	C.pygo_lion_step_f32(
		(*C.float)(param.DataPtr()),
		(*C.float)(grad.DataPtr()),
		(*C.float)(state.M.DataPtr()),
		C.float(lr), C.float(beta1), C.float(beta2), C.float(wd),
		C.int64_t(param.NumEl()),
	)
	return nil
}

// ─── LAMB ────────────────────────────────────────────────────────────────────

// LAMBState holds moment buffers for LAMB.
type LAMBState struct {
	M, V *BridgeTensor
	Step int64
}

// NewLAMBState allocates LAMB moment buffers.
func NewLAMBState(param *BridgeTensor) (*LAMBState, error) {
	m, err := NewBridgeTensor(param.shape, Float32, param.device)
	if err != nil {
		return nil, err
	}
	v, err := NewBridgeTensor(param.shape, Float32, param.device)
	if err != nil {
		return nil, err
	}
	Zeros(m)
	Zeros(v)
	return &LAMBState{M: m, V: v}, nil
}

// LAMBStep performs one LAMB parameter update.
func LAMBStep(
	param, grad *BridgeTensor,
	state *LAMBState,
	lr, beta1, beta2, eps, wd, trustClip float32,
) error {
	state.Step++
	bc1 := float32(1.0 - math.Pow(float64(beta1), float64(state.Step)))
	bc2 := float32(1.0 - math.Pow(float64(beta2), float64(state.Step)))
	C.pygo_lamb_step_f32(
		(*C.float)(param.DataPtr()),
		(*C.float)(grad.DataPtr()),
		(*C.float)(state.M.DataPtr()),
		(*C.float)(state.V.DataPtr()),
		C.float(lr), C.float(beta1), C.float(beta2),
		C.float(eps), C.float(wd),
		C.float(bc1), C.float(bc2), C.float(trustClip),
		C.int64_t(param.NumEl()),
	)
	return nil
}

// ─── Gradient clipping ────────────────────────────────────────────────────────

// ClipGradNorm clips gradients of params to maxNorm. Returns actual norm.
func ClipGradNorm(params []*BridgeTensor, maxNorm float32) (float32, error) {
	if len(params) == 0 {
		return 0, nil
	}
	n := len(params)
	ptrs := make([]*C.float, n)
	sizes := make([]C.int64_t, n)
	for i, p := range params {
		ptrs[i] = (*C.float)(p.DataPtr())
		sizes[i] = C.int64_t(p.NumEl())
	}
	norm := float32(C.pygo_grad_norm_f32(
		(**C.float)(unsafe.Pointer(&ptrs[0])),
		&sizes[0],
		C.int(n),
	))
	if norm > maxNorm {
		C.pygo_clip_grad_norm_f32(
			(**C.float)(unsafe.Pointer(&ptrs[0])),
			&sizes[0],
			C.int(n),
			C.float(maxNorm),
		)
	}
	return norm, nil
}

// ─── EMA ─────────────────────────────────────────────────────────────────────

// EMAUpdate updates shadow = decay*shadow + (1-decay)*param.
func EMAUpdate(shadow, param *BridgeTensor, decay float32) error {
	if shadow.NumEl() != param.NumEl() {
		return fmt.Errorf("EMA: shape mismatch %v vs %v", shadow.shape, param.shape)
	}
	C.pygo_ema_update_f32(
		(*C.float)(shadow.DataPtr()),
		(*C.float)(param.DataPtr()),
		C.float(decay),
		C.int64_t(param.NumEl()),
	)
	return nil
}

// dummyUnsafe avoids unused import
var _dummyUnsafe = unsafe.Sizeof(0)
