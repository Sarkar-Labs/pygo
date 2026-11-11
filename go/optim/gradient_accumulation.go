package optim

import "math"

// GeneratedGradientAccumulation implements a deterministic scalar update corresponding to gradient accumulation.
type GeneratedGradientAccumulation struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedGradientAccumulation(rate float64) *GeneratedGradientAccumulation {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedGradientAccumulation{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedGradientAccumulation) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
