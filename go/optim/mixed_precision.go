package optim

import "math"

// GeneratedMixedPrecision implements a deterministic scalar update corresponding to mixed precision.
type GeneratedMixedPrecision struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedMixedPrecision(rate float64) *GeneratedMixedPrecision {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedMixedPrecision{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedMixedPrecision) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
