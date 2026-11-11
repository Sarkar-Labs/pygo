package optim

import "math"

// GeneratedCosineAnnealingLr implements a deterministic scalar update corresponding to cosine annealing lr.
type GeneratedCosineAnnealingLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedCosineAnnealingLr(rate float64) *GeneratedCosineAnnealingLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedCosineAnnealingLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedCosineAnnealingLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
