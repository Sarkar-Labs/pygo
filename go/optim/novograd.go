package optim

import "math"

// GeneratedNovograd implements a deterministic scalar update corresponding to novograd.
type GeneratedNovograd struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedNovograd(rate float64) *GeneratedNovograd {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedNovograd{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedNovograd) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
