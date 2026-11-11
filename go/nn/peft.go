package nn

import (
	"fmt"
	"github.com/pygo/go/tensor"
)

type IA3Linear struct {
	BaseModule
	base  *Linear
	scale *Parameter
}

func NewIA3Linear(base *Linear) *IA3Linear {
	a := &IA3Linear{BaseModule: NewBase(fmt.Sprintf("IA3(%d->%d)", base.InFeatures, base.OutFeatures)), base: base}
	a.scale = NewParameter(tensor.Ones(tensor.Shape{int64(base.OutFeatures)}))
	a.RegisterParameter("ia3_scale", a.scale)
	return a
}
func (a *IA3Linear) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	return a.base.Forward(inputs[0]).Mul(a.scale.Tensor)
}

type PromptTuning struct {
	BaseModule
	NumTokens  int
	SoftPrompt *Parameter
}

func NewPromptTuning(numTokens, embedDim int) *PromptTuning {
	p := &PromptTuning{BaseModule: NewBase(fmt.Sprintf("PromptTuning(n=%d)", numTokens)), NumTokens: numTokens}
	p.SoftPrompt = NewParameter(tensor.Randn(tensor.Shape{1, int64(numTokens), int64(embedDim)}).MulScalar(0.01))
	p.RegisterParameter("soft_prompt", p.SoftPrompt)
	return p
}
func (p *PromptTuning) PrependPrompt(embeddings *tensor.Tensor) *tensor.Tensor {
	B := embeddings.Dim(0)
	prompt := p.SoftPrompt.Tensor.Expand(B, int64(p.NumTokens), embeddings.Dim(2))
	return tensor.Cat([]*tensor.Tensor{prompt, embeddings}, 1)
}
func (p *PromptTuning) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	return p.PrependPrompt(inputs[0])
}
