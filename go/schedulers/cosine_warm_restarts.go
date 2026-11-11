package schedulers

import "math"

// GeneratedCosineWarmRestarts implements a deterministic scalar update corresponding to cosine warm restarts.
type GeneratedCosineWarmRestarts struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedCosineWarmRestarts(rate float64) *GeneratedCosineWarmRestarts {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedCosineWarmRestarts{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedCosineWarmRestarts) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
