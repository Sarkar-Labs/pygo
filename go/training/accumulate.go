package training

import (
	"github.com/pygo/go/nn"
	"github.com/pygo/go/optim"
)

type GradientAccumulator struct {
	Model       nn.Module
	Optimizer   optim.Optimizer
	AccumSteps  int
	MaxGradNorm float64
	currentStep int
}

func NewGradAccumulator(model nn.Module, opt optim.Optimizer, steps int, maxNorm float64) *GradientAccumulator {
	return &GradientAccumulator{Model: model, Optimizer: opt, AccumSteps: steps, MaxGradNorm: maxNorm}
}
func (ga *GradientAccumulator) ShouldStep() bool {
	ga.currentStep++
	return ga.currentStep%ga.AccumSteps == 0
}
func (ga *GradientAccumulator) Step() float64 {
	params := ga.Model.Parameters()
	for _, p := range params {
		if g := p.Grad(); g != nil {
			_ = g.MulScalar(1.0 / float64(ga.AccumSteps))
		}
	}
	norm := 0.0
	for _, p := range params {
		if g := p.Grad(); g != nil {
			n := g.Pow(2).Sum(-1, false).Sqrt().Item()
			norm += n * n
		}
	}
	if ga.MaxGradNorm > 0 && norm > ga.MaxGradNorm {
		coef := ga.MaxGradNorm / (norm + 1e-6)
		for _, p := range params {
			if g := p.Grad(); g != nil {
				_ = g.MulScalar(coef)
			}
		}
	}
	ga.Optimizer.Step()
	ga.Optimizer.ZeroGrad()
	return norm
}
func (ga *GradientAccumulator) ZeroGrad() { ga.Optimizer.ZeroGrad() }
