// Package models provides pretrained model implementations for PyGo.
package models

import (
	"fmt"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
	"math"
)

// ─── LLaMA Config ────────────────────────────────────────────────────────────

type LLaMAConfig struct {
	VocabSize             int     `json:"vocab_size"`
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumKVHeads            int     `json:"num_key_value_heads"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	RoPETheta             float64 `json:"rope_theta"`
	RMSNormEps            float64 `json:"rms_norm_eps"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`
	TorchDtype            string  `json:"torch_dtype"`
	ArchType              string  `json:"architectures"`
}

func LLaMA3_8B() LLaMAConfig {
	return LLaMAConfig{
		VocabSize: 128256, HiddenSize: 4096, IntermediateSize: 14336,
		NumHiddenLayers: 32, NumAttentionHeads: 32, NumKVHeads: 8,
		MaxPositionEmbeddings: 131072, RoPETheta: 500000.0,
		RMSNormEps: 1e-5, TieWordEmbeddings: false, TorchDtype: "bfloat16",
	}
}

func LLaMA3_70B() LLaMAConfig {
	return LLaMAConfig{
		VocabSize: 128256, HiddenSize: 8192, IntermediateSize: 28672,
		NumHiddenLayers: 80, NumAttentionHeads: 64, NumKVHeads: 8,
		MaxPositionEmbeddings: 131072, RoPETheta: 500000.0,
		RMSNormEps: 1e-5, TieWordEmbeddings: false, TorchDtype: "bfloat16",
	}
}

func Mistral7B() LLaMAConfig {
	return LLaMAConfig{
		VocabSize: 32000, HiddenSize: 4096, IntermediateSize: 14336,
		NumHiddenLayers: 32, NumAttentionHeads: 32, NumKVHeads: 8,
		MaxPositionEmbeddings: 32768, RoPETheta: 10000.0,
		RMSNormEps: 1e-5, TieWordEmbeddings: false, TorchDtype: "bfloat16",
	}
}

// ─── RMSNorm ─────────────────────────────────────────────────────────────────

type RMSNorm struct {
	nn.BaseModule
	Weight *nn.Parameter
	Eps    float64
	Dim    int
}

func NewRMSNorm(dim int, eps float64) *RMSNorm {
	r := &RMSNorm{
		BaseModule: nn.NewBase(fmt.Sprintf("RMSNorm(%d)", dim)),
		Eps:        eps, Dim: dim,
	}
	r.Weight = nn.NewParameter(tensor.Ones(tensor.Shape{int64(dim)}))
	r.RegisterParameter("weight", r.Weight)
	return r
}

func (r *RMSNorm) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	// x * rsqrt(mean(x^2) + eps) * weight
	rms := x.Pow(2).Mean(-1, true).AddScalar(r.Eps).Sqrt()
	return x.Div(rms).Mul(r.Weight.Tensor)
}

// ─── RotaryEmbedding ─────────────────────────────────────────────────────────

type RotaryEmbedding struct {
	nn.BaseModule
	Dim    int
	MaxSeq int
	Theta  float64
}

func NewRotaryEmbedding(dim, maxSeq int, theta float64) *RotaryEmbedding {
	return &RotaryEmbedding{
		BaseModule: nn.NewBase("RotaryEmbedding"),
		Dim:        dim, MaxSeq: maxSeq, Theta: theta,
	}
}

func (r *RotaryEmbedding) buildFreqs(device tensor.Device) (*tensor.Tensor, *tensor.Tensor) {
	half := int64(r.Dim / 2)
	// inv_freq[i] = 1 / theta^(2i/dim)
	invFreqData := make([]float32, half)
	for i := int64(0); i < half; i++ {
		invFreqData[i] = float32(1.0 / math.Pow(r.Theta, float64(2*i)/float64(r.Dim)))
	}
	invFreq := tensor.FromSlice(invFreqData, tensor.Shape{half}, tensor.WithDevice(device))
	t := tensor.Arange(0, float64(r.MaxSeq), 1, tensor.WithDevice(device))
	// freqs = outer(t, invFreq) [MaxSeq, Dim/2]
	freqs := t.Unsqueeze(1).MatMul(invFreq.Unsqueeze(0))
	emb := tensor.Cat([]*tensor.Tensor{freqs, freqs}, -1) // [MaxSeq, Dim]
	return emb.Cos(), emb.Sin()
}

func (r *RotaryEmbedding) Apply(q, k *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	cos, sin := r.buildFreqs(q.Device())
	seqLen := q.Dim(2)
	cos = cos.Slice(0, 0, seqLen, 1).Unsqueeze(0).Unsqueeze(0)
	sin = sin.Slice(0, 0, seqLen, 1).Unsqueeze(0).Unsqueeze(0)

	// rotate_half
	rotateHalf := func(x *tensor.Tensor) *tensor.Tensor {
		half := x.Dim(-1) / 2
		x1 := x.Slice(-1, 0, half, 1)
		x2 := x.Slice(-1, half, x.Dim(-1), 1)
		return tensor.Cat([]*tensor.Tensor{x2.Neg(), x1}, -1)
	}

	qRot := q.Mul(cos).Add(rotateHalf(q).Mul(sin))
	kRot := k.Mul(cos).Add(rotateHalf(k).Mul(sin))
	return qRot, kRot
}

func (r *RotaryEmbedding) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	panic("Use Apply() instead")
}

// ─── GQA Attention ───────────────────────────────────────────────────────────

type LLaMAAttention struct {
	nn.BaseModule
	Cfg     LLaMAConfig
	QProjW  *nn.Parameter
	KProjW  *nn.Parameter
	VProjW  *nn.Parameter
	OProjW  *nn.Parameter
	Rope    *RotaryEmbedding
	HeadDim int
	NRep    int // q heads per kv head
}

func NewLLaMAAttention(cfg LLaMAConfig) *LLaMAAttention {
	headDim := cfg.HiddenSize / cfg.NumAttentionHeads
	a := &LLaMAAttention{
		BaseModule: nn.NewBase("LLaMAAttention"),
		Cfg:        cfg, HeadDim: headDim,
		NRep: cfg.NumAttentionHeads / cfg.NumKVHeads,
	}

	newLinearWeight := func(outF, inF int) *nn.Parameter {
		w := tensor.Randn(tensor.Shape{int64(outF), int64(inF)}).MulScalar(0.02)
		return nn.NewParameter(w)
	}

	a.QProjW = newLinearWeight(cfg.NumAttentionHeads*headDim, cfg.HiddenSize)
	a.KProjW = newLinearWeight(cfg.NumKVHeads*headDim, cfg.HiddenSize)
	a.VProjW = newLinearWeight(cfg.NumKVHeads*headDim, cfg.HiddenSize)
	a.OProjW = newLinearWeight(cfg.HiddenSize, cfg.NumAttentionHeads*headDim)
	a.Rope = NewRotaryEmbedding(headDim, cfg.MaxPositionEmbeddings, cfg.RoPETheta)

	a.RegisterParameter("q_proj.weight", a.QProjW)
	a.RegisterParameter("k_proj.weight", a.KProjW)
	a.RegisterParameter("v_proj.weight", a.VProjW)
	a.RegisterParameter("o_proj.weight", a.OProjW)
	a.RegisterModule("rotary_emb", a.Rope)
	return a
}

func (a *LLaMAAttention) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	B, S, _ := x.Dim(0), x.Dim(1), x.Dim(2)

	proj := func(w *nn.Parameter) *tensor.Tensor {
		return x.MatMul(w.Transpose(-2, -1))
	}

	nH := int64(a.Cfg.NumAttentionHeads)
	nKV := int64(a.Cfg.NumKVHeads)
	d := int64(a.HeadDim)

	q := proj(a.QProjW).Reshape(tensor.Shape{B, S, nH, d}).Permute([]int{0, 2, 1, 3})
	k := proj(a.KProjW).Reshape(tensor.Shape{B, S, nKV, d}).Permute([]int{0, 2, 1, 3})
	v := proj(a.VProjW).Reshape(tensor.Shape{B, S, nKV, d}).Permute([]int{0, 2, 1, 3})

	q, k = a.Rope.Apply(q, k)

	// GQA: repeat kv heads
	if a.NRep > 1 {
		k = k.Unsqueeze(2).Repeat([]int{1, 1, a.NRep, 1, 1}).Reshape(tensor.Shape{B, nH, -1, d})
		v = v.Unsqueeze(2).Repeat([]int{1, 1, a.NRep, 1, 1}).Reshape(tensor.Shape{B, nH, -1, d})
	}

	scale := float64(1.0 / math.Sqrt(float64(a.HeadDim)))
	scores := q.MatMul(k.Transpose(-2, -1)).MulScalar(scale)
	// Causal mask
	scores = applyCausalMask(scores)
	attn := softmaxTensor(scores, -1)
	out := attn.MatMul(v)

	out = out.Permute([]int{0, 2, 1, 3}).Reshape(tensor.Shape{B, S, nH * d})
	return out.MatMul(a.OProjW.Transpose(-2, -1))
}

func applyCausalMask(scores *tensor.Tensor) *tensor.Tensor {
	S := scores.Dim(-1)
	mask := tensor.Ones(tensor.Shape{S, S}).MulScalar(math.Inf(-1))
	mask = mask.Triu(1).Unsqueeze(0).Unsqueeze(0)
	return scores.Add(mask)
}

func softmaxTensor(x *tensor.Tensor, dim int) *tensor.Tensor {
	return x.Softmax(dim)
}

// ─── SwiGLU MLP ──────────────────────────────────────────────────────────────

type LLaMAMLP struct {
	nn.BaseModule
	GateW *nn.Parameter
	UpW   *nn.Parameter
	DownW *nn.Parameter
}

func NewLLaMAMLP(cfg LLaMAConfig) *LLaMAMLP {
	m := &LLaMAMLP{BaseModule: nn.NewBase("LLaMAMLP")}
	newW := func(out, in int) *nn.Parameter {
		w := tensor.Randn(tensor.Shape{int64(out), int64(in)}).MulScalar(0.02)
		return nn.NewParameter(w)
	}
	m.GateW = newW(cfg.IntermediateSize, cfg.HiddenSize)
	m.UpW = newW(cfg.IntermediateSize, cfg.HiddenSize)
	m.DownW = newW(cfg.HiddenSize, cfg.IntermediateSize)
	m.RegisterParameter("gate_proj.weight", m.GateW)
	m.RegisterParameter("up_proj.weight", m.UpW)
	m.RegisterParameter("down_proj.weight", m.DownW)
	return m
}

func (m *LLaMAMLP) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	gate := x.MatMul(m.GateW.Transpose(-2, -1)).SiLU()
	up := x.MatMul(m.UpW.Transpose(-2, -1))
	return gate.Mul(up).MatMul(m.DownW.Transpose(-2, -1))
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

type LLaMADecoderLayer struct {
	nn.BaseModule
	SelfAttn  *LLaMAAttention
	MLP       *LLaMAMLP
	InputNorm *RMSNorm
	PostNorm  *RMSNorm
}

func NewLLaMADecoderLayer(cfg LLaMAConfig) *LLaMADecoderLayer {
	l := &LLaMADecoderLayer{BaseModule: nn.NewBase("LLaMADecoderLayer")}
	l.SelfAttn = NewLLaMAAttention(cfg)
	l.MLP = NewLLaMAMLP(cfg)
	l.InputNorm = NewRMSNorm(cfg.HiddenSize, cfg.RMSNormEps)
	l.PostNorm = NewRMSNorm(cfg.HiddenSize, cfg.RMSNormEps)
	l.RegisterModule("self_attn", l.SelfAttn)
	l.RegisterModule("mlp", l.MLP)
	l.RegisterModule("input_layernorm", l.InputNorm)
	l.RegisterModule("post_attention_layernorm", l.PostNorm)
	return l
}

func (l *LLaMADecoderLayer) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	// Pre-norm attention with residual
	residual := x
	x = l.InputNorm.Forward(x)
	x = l.SelfAttn.Forward(x)
	x = x.Add(residual)
	// Pre-norm FFN with residual
	residual = x
	x = l.MLP.Forward(l.PostNorm.Forward(x))
	return x.Add(residual)
}

// ─── LLaMA Model ─────────────────────────────────────────────────────────────

type LLaMAModel struct {
	nn.BaseModule
	Cfg    LLaMAConfig
	Embed  *nn.Embedding
	Layers []*LLaMADecoderLayer
	Norm   *RMSNorm
	LMHead *nn.Parameter
}

func NewLLaMA(cfg LLaMAConfig) *LLaMAModel {
	m := &LLaMAModel{
		BaseModule: nn.NewBase(fmt.Sprintf("LLaMA(%dL,%dH)", cfg.NumHiddenLayers, cfg.NumAttentionHeads)),
		Cfg:        cfg,
	}
	m.Embed = nn.NewEmbedding(cfg.VocabSize, cfg.HiddenSize, -1)
	m.Norm = NewRMSNorm(cfg.HiddenSize, cfg.RMSNormEps)
	m.LMHead = nn.NewParameter(tensor.Randn(tensor.Shape{int64(cfg.VocabSize), int64(cfg.HiddenSize)}).MulScalar(0.02))

	m.Layers = make([]*LLaMADecoderLayer, cfg.NumHiddenLayers)
	for i := range m.Layers {
		m.Layers[i] = NewLLaMADecoderLayer(cfg)
		m.RegisterModule(fmt.Sprintf("layers.%d", i), m.Layers[i])
	}
	m.RegisterModule("embed_tokens", m.Embed)
	m.RegisterModule("norm", m.Norm)
	m.RegisterParameter("lm_head.weight", m.LMHead)
	return m
}

func (m *LLaMAModel) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	inputIDs := inputs[0] // [B, S]
	x := m.Embed.Forward(inputIDs)
	for _, layer := range m.Layers {
		x = layer.Forward(x)
	}
	x = m.Norm.Forward(x)
	return x.MatMul(m.LMHead.Transpose(-2, -1)) // [B, S, V]
}

func (m *LLaMAModel) NumParams() (total, trainable int64) {
	return nn.CountParams(m)
}

func (m *LLaMAModel) String() string {
	total, _ := m.NumParams()
	return fmt.Sprintf("LLaMA(layers=%d, heads=%d, kv_heads=%d, hidden=%d, vocab=%d, params=%.2fB)",
		m.Cfg.NumHiddenLayers, m.Cfg.NumAttentionHeads, m.Cfg.NumKVHeads,
		m.Cfg.HiddenSize, m.Cfg.VocabSize, float64(total)/1e9)
}

// ─── Factory ─────────────────────────────────────────────────────────────────

func LLaMA3_8BModel() *LLaMAModel  { return NewLLaMA(LLaMA3_8B()) }
func LLaMA3_70BModel() *LLaMAModel { return NewLLaMA(LLaMA3_70B()) }
func Mistral7BModel() *LLaMAModel  { return NewLLaMA(Mistral7B()) }
