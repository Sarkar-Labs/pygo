package optim

import "math"

// GeneratedAdamax implements a deterministic scalar update corresponding to adamax.
type GeneratedAdamax struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedAdamax(rate float64) *GeneratedAdamax {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedAdamax{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedAdamax) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
