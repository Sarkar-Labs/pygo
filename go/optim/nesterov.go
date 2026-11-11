package optim

import "math"

// GeneratedNesterov implements a deterministic scalar update corresponding to nesterov.
type GeneratedNesterov struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedNesterov(rate float64) *GeneratedNesterov {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedNesterov{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedNesterov) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
