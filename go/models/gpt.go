package models

import (
	"fmt"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
)

type GPTConfig struct {
	VocabSize, MaxSeqLen, EmbedDim, NumHeads, NumLayers int
	MLPRatio, Dropout                                   float64
}

func GPT2Small() GPTConfig  { return GPTConfig{50257, 1024, 768, 12, 12, 4., 0.1} }
func GPT2Medium() GPTConfig { return GPTConfig{50257, 1024, 1024, 16, 24, 4., 0.1} }
func GPT2Large() GPTConfig  { return GPTConfig{50257, 1024, 1280, 20, 36, 4., 0.1} }
func GPT2XL() GPTConfig     { return GPTConfig{50257, 1024, 1600, 25, 48, 4., 0.1} }

type GPTBlock struct {
	nn.BaseModule
	LN1, LN2 *nn.LayerNorm
	Attn     *nn.MultiHeadAttention
	MLP      *nn.Sequential
	Drop     *nn.Dropout
}

func NewGPTBlock(cfg GPTConfig) *GPTBlock {
	b := &GPTBlock{BaseModule: nn.NewBase("GPTBlock")}
	ff := int(float64(cfg.EmbedDim) * cfg.MLPRatio)
	b.LN1 = nn.NewLayerNorm([]int{cfg.EmbedDim}, 1e-5)
	b.Attn = nn.NewMultiHeadAttention(cfg.EmbedDim, cfg.NumHeads, 0)
	b.LN2 = nn.NewLayerNorm([]int{cfg.EmbedDim}, 1e-5)
	b.MLP = nn.NewSequential(nn.NewLinear(cfg.EmbedDim, ff, true), nn.NewGELU(), nn.NewDropout(cfg.Dropout), nn.NewLinear(ff, cfg.EmbedDim, true))
	b.Drop = nn.NewDropout(cfg.Dropout)
	b.RegisterModule("ln_1", b.LN1)
	b.RegisterModule("attn", b.Attn)
	b.RegisterModule("ln_2", b.LN2)
	b.RegisterModule("mlp", b.MLP)
	return b
}
func (b *GPTBlock) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	x = x.Add(b.Drop.Forward(b.Attn.Forward(b.LN1.Forward(x))))
	return x.Add(b.Drop.Forward(b.MLP.Forward(b.LN2.Forward(x))))
}

type GPT struct {
	nn.BaseModule
	Cfg              GPTConfig
	TokenEmb, PosEmb *nn.Embedding
	Drop             *nn.Dropout
	Layers           []*GPTBlock
	LNF              *nn.LayerNorm
	LMHead           *nn.Parameter
}

func NewGPT(cfg GPTConfig) *GPT {
	g := &GPT{BaseModule: nn.NewBase(fmt.Sprintf("GPT(%dL,%dH)", cfg.NumLayers, cfg.NumHeads)), Cfg: cfg}
	g.TokenEmb = nn.NewEmbedding(cfg.VocabSize, cfg.EmbedDim)
	g.PosEmb = nn.NewEmbedding(cfg.MaxSeqLen, cfg.EmbedDim)
	g.Drop = nn.NewDropout(cfg.Dropout)
	g.LNF = nn.NewLayerNorm([]int{cfg.EmbedDim}, 1e-5)
	g.LMHead = nn.NewParameter(tensor.Randn(tensor.Shape{int64(cfg.VocabSize), int64(cfg.EmbedDim)}).MulScalar(0.02))
	g.Layers = make([]*GPTBlock, cfg.NumLayers)
	for i := range g.Layers {
		g.Layers[i] = NewGPTBlock(cfg)
		g.RegisterModule(fmt.Sprintf("h.%d", i), g.Layers[i])
	}
	g.RegisterModule("wte", g.TokenEmb)
	g.RegisterModule("wpe", g.PosEmb)
	g.RegisterModule("ln_f", g.LNF)
	return g
}
func (g *GPT) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	ids := inputs[0]
	S := ids.Dim(1)
	pos := tensor.Arange(0, float64(S), 1).Unsqueeze(0)
	x := g.Drop.Forward(g.TokenEmb.Forward(ids).Add(g.PosEmb.Forward(pos)))
	for _, l := range g.Layers {
		x = l.Forward(x)
	}
	return g.LNF.Forward(x).MatMul(g.LMHead.Tensor.Transpose(-2, -1))
}
func GPT2SmallModel() *GPT  { return NewGPT(GPT2Small()) }
func GPT2MediumModel() *GPT { return NewGPT(GPT2Medium()) }
func GPT2LargeModel() *GPT  { return NewGPT(GPT2Large()) }
func GPT2XLModel() *GPT     { return NewGPT(GPT2XL()) }
