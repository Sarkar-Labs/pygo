package optim

import "math"

// GeneratedAdamw implements a deterministic scalar update corresponding to adamw.
type GeneratedAdamw struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedAdamw(rate float64) *GeneratedAdamw {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedAdamw{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedAdamw) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
