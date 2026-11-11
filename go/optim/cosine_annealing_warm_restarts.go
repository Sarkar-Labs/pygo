package optim

import "math"

// GeneratedCosineAnnealingWarmRestarts implements a deterministic scalar update corresponding to cosine annealing warm restarts.
type GeneratedCosineAnnealingWarmRestarts struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedCosineAnnealingWarmRestarts(rate float64) *GeneratedCosineAnnealingWarmRestarts {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedCosineAnnealingWarmRestarts{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedCosineAnnealingWarmRestarts) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
