// nn_cgo.go — Neural network forward passes implemented in C via CGo.
package cgo

/*
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// ─── Linear (fully-connected) ─────────────────────────────────────────────────
// out[B,O] = x[B,I] @ W[O,I]^T + b[O]

static void pygo_linear_f32(
    float* x, float* W, float* b, float* out,
    int64_t B, int64_t I, int64_t O
) {
    for (int64_t bi = 0; bi < B; bi++) {
        for (int64_t o = 0; o < O; o++) {
            float acc = b ? b[o] : 0.f;
            for (int64_t i = 0; i < I; i++)
                acc += x[bi*I+i] * W[o*I+i];
            out[bi*O+o] = acc;
        }
    }
}

// ─── Embedding lookup ─────────────────────────────────────────────────────────
// out[B,S,D] = emb[ids[B,S], :]

static void pygo_embedding_f32(
    float* emb, int64_t* ids, float* out,
    int64_t B, int64_t S, int64_t D
) {
    for (int64_t b = 0; b < B; b++)
        for (int64_t s = 0; s < S; s++) {
            int64_t idx = ids[b*S+s];
            float*  src = emb + idx*D;
            float*  dst = out + (b*S+s)*D;
            memcpy(dst, src, D * sizeof(float));
        }
}

// ─── Scaled dot-product attention ─────────────────────────────────────────────
// Q,K,V: [B,H,S,D]  out: [B,H,S,D]  scale=1/sqrt(D)

static void pygo_sdp_attention_f32(
    float* Q, float* K, float* V, float* out,
    int64_t B, int64_t H, int64_t S, int64_t D,
    float scale, int causal
) {
    int64_t HS  = H*S*D;
    float* scores = (float*)malloc(S*S*sizeof(float));
    if (!scores) return;

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            float* q = Q + b*HS + h*S*D;
            float* k = K + b*HS + h*S*D;
            float* v = V + b*HS + h*S*D;
            float* o = out + b*HS + h*S*D;

            // scores[i,j] = Q[i,:] · K[j,:] * scale
            for (int64_t i = 0; i < S; i++) {
                for (int64_t j = 0; j < S; j++) {
                    if (causal && j > i) { scores[i*S+j] = -1e9f; continue; }
                    float acc = 0.f;
                    for (int64_t d = 0; d < D; d++)
                        acc += q[i*D+d] * k[j*D+d];
                    scores[i*S+j] = acc * scale;
                }
            }

            // softmax over last dim
            for (int64_t i = 0; i < S; i++) {
                float mx = scores[i*S];
                for (int64_t j = 1; j < S; j++) if (scores[i*S+j] > mx) mx = scores[i*S+j];
                float s = 0.f;
                for (int64_t j = 0; j < S; j++) { scores[i*S+j] = expf(scores[i*S+j]-mx); s += scores[i*S+j]; }
                if (s > 0.f) for (int64_t j = 0; j < S; j++) scores[i*S+j] /= s;
            }

            // out[i,:] = sum_j scores[i,j] * V[j,:]
            for (int64_t i = 0; i < S; i++) {
                memset(o + i*D, 0, D*sizeof(float));
                for (int64_t j = 0; j < S; j++)
                    for (int64_t d = 0; d < D; d++)
                        o[i*D+d] += scores[i*S+j] * v[j*D+d];
            }
        }
    }
    free(scores);
}

// ─── Conv2d (naive, CPU reference) ────────────────────────────────────────────
// x:[B,C,H,W]  W:[OC,IC,KH,KW]  out:[B,OC,OH,OW]

static void pygo_conv2d_f32(
    float* x, float* W, float* b, float* out,
    int64_t B,int64_t IC,int64_t H,int64_t W_dim,
    int64_t OC,int64_t KH,int64_t KW,
    int64_t stride,int64_t pad
) {
    int64_t OH = (H + 2*pad - KH) / stride + 1;
    int64_t OW = (W_dim + 2*pad - KW) / stride + 1;
    for (int64_t bn = 0; bn < B;  bn++)
    for (int64_t oc = 0; oc < OC; oc++) {
        for (int64_t oh = 0; oh < OH; oh++)
        for (int64_t ow = 0; ow < OW; ow++) {
            float acc = b ? b[oc] : 0.f;
            for (int64_t ic = 0; ic < IC; ic++)
            for (int64_t kh = 0; kh < KH; kh++)
            for (int64_t kw = 0; kw < KW; kw++) {
                int64_t ih = oh*stride - pad + kh;
                int64_t iw = ow*stride - pad + kw;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W_dim) continue;
                acc += x[bn*IC*H*W_dim + ic*H*W_dim + ih*W_dim + iw]
                     * W[oc*IC*KH*KW   + ic*KH*KW   + kh*KW    + kw];
            }
            out[bn*OC*OH*OW + oc*OH*OW + oh*OW + ow] = acc;
        }
    }
}

// ─── MaxPool2d ────────────────────────────────────────────────────────────────

static void pygo_maxpool2d_f32(
    float* x, float* out, int64_t* idx,
    int64_t B,int64_t C,int64_t H,int64_t W,
    int64_t KH,int64_t KW,int64_t stride,int64_t pad
) {
    int64_t OH = (H + 2*pad - KH) / stride + 1;
    int64_t OW = (W + 2*pad - KW) / stride + 1;
    for (int64_t b = 0; b < B; b++)
    for (int64_t c = 0; c < C; c++)
    for (int64_t oh = 0; oh < OH; oh++)
    for (int64_t ow = 0; ow < OW; ow++) {
        float mx = -1e38f; int64_t best = -1;
        for (int64_t kh = 0; kh < KH; kh++)
        for (int64_t kw = 0; kw < KW; kw++) {
            int64_t ih = oh*stride - pad + kh;
            int64_t iw = ow*stride - pad + kw;
            if (ih < 0||ih >= H||iw < 0||iw >= W) continue;
            int64_t xi = b*C*H*W + c*H*W + ih*W + iw;
            if (x[xi] > mx) { mx = x[xi]; best = xi; }
        }
        out[b*C*OH*OW + c*OH*OW + oh*OW + ow] = mx;
        if (idx) idx[b*C*OH*OW + c*OH*OW + oh*OW + ow] = best;
    }
}

// ─── AdaptiveAvgPool2d → [B,C,1,1] ──────────────────────────────────────────

static void pygo_adaptive_avg_pool2d_f32(
    float* x, float* out,
    int64_t B, int64_t C, int64_t H, int64_t W
) {
    float inv = 1.f / (float)(H*W);
    for (int64_t b = 0; b < B; b++)
    for (int64_t c = 0; c < C; c++) {
        float s = 0.f;
        for (int64_t h = 0; h < H; h++)
        for (int64_t w = 0; w < W; w++)
            s += x[b*C*H*W + c*H*W + h*W + w];
        out[b*C + c] = s * inv;
    }
}

// ─── Dropout (training) ───────────────────────────────────────────────────────

static void pygo_dropout_f32(
    float* x, float* out, uint8_t* mask,
    int64_t n, float p, uint32_t seed
) {
    uint32_t s = seed;
    float scale = p < 1.f ? 1.f/(1.f-p) : 0.f;
    for (int64_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5; // xorshift
        float u = (s & 0x7fffff) / (float)0x800000;
        mask[i] = u > p ? 1 : 0;
        out[i]  = mask[i] ? x[i]*scale : 0.f;
    }
}

// ─── GeLU approximate (tanh variant) ─────────────────────────────────────────

static void pygo_gelu_tanh_f32(float* x, float* out, int64_t n) {
    const float c  = 0.7978845608f;
    const float c2 = 0.044715f;
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        float inner = c * (v + c2*v*v*v);
        out[i] = 0.5f * v * (1.f + tanhf(inner));
    }
}
*/
import "C"
import (
	"fmt"
	"math"
	"unsafe"
)

// ─── Linear ───────────────────────────────────────────────────────────────────

// Linear computes out = x @ W^T + b.
// x: [B,I], W: [O,I], b: [O] or nil, out: [B,O]
func Linear(x, W, b, out *BridgeTensor) error {
	if len(x.shape) < 2 {
		return fmt.Errorf("Linear: x must be ≥2D, got %v", x.shape)
	}
	B := x.NumEl() / x.shape[len(x.shape)-1]
	I := x.shape[len(x.shape)-1]
	O := W.shape[0]
	var bp *C.float
	if b != nil {
		bp = (*C.float)(b.DataPtr())
	}
	C.pygo_linear_f32(
		(*C.float)(x.DataPtr()),
		(*C.float)(W.DataPtr()),
		bp,
		(*C.float)(out.DataPtr()),
		C.int64_t(B), C.int64_t(I), C.int64_t(O),
	)
	return nil
}

// ─── Embedding ────────────────────────────────────────────────────────────────

// EmbeddingLookup gathers rows from emb table.
// emb: [VocabSize, D], ids: [B,S] int64, out: [B,S,D]
func EmbeddingLookup(emb *BridgeTensor, ids *BridgeTensor, out *BridgeTensor) error {
	if len(ids.shape) < 2 {
		return fmt.Errorf("EmbeddingLookup: ids must be 2D [B,S]")
	}
	B := ids.shape[0]
	S := ids.shape[1]
	D := emb.shape[1]
	C.pygo_embedding_f32(
		(*C.float)(emb.DataPtr()),
		(*C.int64_t)(ids.DataPtr()),
		(*C.float)(out.DataPtr()),
		C.int64_t(B), C.int64_t(S), C.int64_t(D),
	)
	return nil
}

// ─── Scaled Dot-Product Attention ─────────────────────────────────────────────

// ScaledDotProductAttention computes attention(Q,K,V).
// Q,K,V,out: [B,H,S,D]
// causal=true enables causal masking (for decoder/autoregressive models)
func ScaledDotProductAttention(Q, K, V, out *BridgeTensor, causal bool) error {
	if len(Q.shape) != 4 {
		return fmt.Errorf("SDP attention: expected 4D [B,H,S,D], got %v", Q.shape)
	}
	B, H, S, D := Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3]
	scale := float32(1.0 / math.Sqrt(float64(D)))
	causalInt := 0
	if causal {
		causalInt = 1
	}
	C.pygo_sdp_attention_f32(
		(*C.float)(Q.DataPtr()),
		(*C.float)(K.DataPtr()),
		(*C.float)(V.DataPtr()),
		(*C.float)(out.DataPtr()),
		C.int64_t(B), C.int64_t(H), C.int64_t(S), C.int64_t(D),
		C.float(scale), C.int(causalInt),
	)
	return nil
}

// ─── Conv2d ───────────────────────────────────────────────────────────────────

// Conv2dForward computes a 2D convolution.
// x: [B,IC,H,W], weight: [OC,IC,KH,KW], bias: [OC] or nil
// out: [B,OC,OH,OW]
func Conv2dForward(x, weight, bias, out *BridgeTensor, stride, pad int) error {
	if len(x.shape) != 4 || len(weight.shape) != 4 {
		return fmt.Errorf("Conv2d: expected 4D tensors")
	}
	B, IC, H, W := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	OC, KH, KW := weight.shape[0], weight.shape[2], weight.shape[3]
	var bp *C.float
	if bias != nil {
		bp = (*C.float)(bias.DataPtr())
	}
	C.pygo_conv2d_f32(
		(*C.float)(x.DataPtr()),
		(*C.float)(weight.DataPtr()),
		bp,
		(*C.float)(out.DataPtr()),
		C.int64_t(B), C.int64_t(IC), C.int64_t(H), C.int64_t(W),
		C.int64_t(OC), C.int64_t(KH), C.int64_t(KW),
		C.int64_t(stride), C.int64_t(pad),
	)
	return nil
}

// ─── MaxPool2d ────────────────────────────────────────────────────────────────

// MaxPool2dForward computes 2D max pooling.
// x: [B,C,H,W], out: [B,C,OH,OW], indices: [B,C,OH,OW] int64 or nil
func MaxPool2dForward(x, out, indices *BridgeTensor, kH, kW, stride, pad int) error {
	if len(x.shape) != 4 {
		return fmt.Errorf("MaxPool2d: expected 4D input")
	}
	B, Ch, H, W := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	var idxp *C.int64_t
	if indices != nil {
		idxp = (*C.int64_t)(indices.DataPtr())
	}
	C.pygo_maxpool2d_f32(
		(*C.float)(x.DataPtr()),
		(*C.float)(out.DataPtr()),
		idxp,
		C.int64_t(B), C.int64_t(Ch), C.int64_t(H), C.int64_t(W),
		C.int64_t(kH), C.int64_t(kW),
		C.int64_t(stride), C.int64_t(pad),
	)
	return nil
}

// ─── AdaptiveAvgPool2d ────────────────────────────────────────────────────────

// AdaptiveAvgPool2d collapses spatial dims to 1x1.
// x: [B,C,H,W], out: [B,C] (reshaped from [B,C,1,1])
func AdaptiveAvgPool2d(x, out *BridgeTensor) error {
	if len(x.shape) != 4 {
		return fmt.Errorf("AdaptiveAvgPool2d: expected 4D input")
	}
	B, Ch, H, W := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	C.pygo_adaptive_avg_pool2d_f32(
		(*C.float)(x.DataPtr()),
		(*C.float)(out.DataPtr()),
		C.int64_t(B), C.int64_t(Ch), C.int64_t(H), C.int64_t(W),
	)
	return nil
}

// ─── Dropout ─────────────────────────────────────────────────────────────────

// DropoutForward applies dropout in training mode.
// mask must be []uint8 of length n (reused for backward).
func DropoutForward(x, out *BridgeTensor, maskBuf []byte, p float32, seed uint32) error {
	n := x.NumEl()
	if int64(len(maskBuf)) < n {
		return fmt.Errorf("Dropout: mask buffer too small (%d < %d)", len(maskBuf), n)
	}
	C.pygo_dropout_f32(
		(*C.float)(x.DataPtr()),
		(*C.float)(out.DataPtr()),
		(*C.uint8_t)(unsafe.Pointer(&maskBuf[0])),
		C.int64_t(n), C.float(p), C.uint32_t(seed),
	)
	return nil
}

// ─── GELU (tanh approximation) ────────────────────────────────────────────────

// GELUTanh computes GELU using tanh approximation (used in GPT-2).
func GELUTanh(x, out *BridgeTensor) error {
	C.pygo_gelu_tanh_f32(
		(*C.float)(x.DataPtr()),
		(*C.float)(out.DataPtr()),
		C.int64_t(x.NumEl()),
	)
	return nil
}

// ─── High-level block helpers ─────────────────────────────────────────────────

// TransformerFFN computes: out = Linear2(GELU(Linear1(x)))
// x: [B,S,D], w1: [FF,D], w2: [D,FF]
func TransformerFFN(x, w1, b1, w2, b2, tmp, out *BridgeTensor) error {
	if err := Linear(x, w1, b1, tmp); err != nil {
		return err
	}
	if err := GELUTanh(tmp, tmp); err != nil {
		return err
	}
	return Linear(tmp, w2, b2, out)
}

// SwiGLUFFN computes: out = Linear3(SiLU(Linear1(x)) * Linear2(x))
func SwiGLUFFN(x, wGate, wUp, wDown, gate, up, out *BridgeTensor) error {
	if err := Linear(x, wGate, nil, gate); err != nil {
		return err
	}
	if err := Linear(x, wUp, nil, up); err != nil {
		return err
	}
	if err := SiLU(gate, gate); err != nil {
		return err
	}
	if err := Mul(gate, up, gate); err != nil {
		return err
	}
	return Linear(gate, wDown, nil, out)
}

// dummyCFloat prevents unused import warning
var _ = (*C.float)(nil)
