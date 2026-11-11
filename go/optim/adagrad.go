package optim

import "math"

// GeneratedAdagrad implements a deterministic scalar update corresponding to adagrad.
type GeneratedAdagrad struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedAdagrad(rate float64) *GeneratedAdagrad {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedAdagrad{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedAdagrad) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
