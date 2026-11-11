package optim

import "math"

// GeneratedMomentum implements a deterministic scalar update corresponding to momentum.
type GeneratedMomentum struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedMomentum(rate float64) *GeneratedMomentum {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedMomentum{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedMomentum) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
