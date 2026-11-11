package optim

import "math"

// GeneratedAmsgrad implements a deterministic scalar update corresponding to amsgrad.
type GeneratedAmsgrad struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedAmsgrad(rate float64) *GeneratedAmsgrad {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedAmsgrad{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedAmsgrad) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
