package optim

import "math"

// GeneratedApollo implements a deterministic scalar update corresponding to apollo.
type GeneratedApollo struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedApollo(rate float64) *GeneratedApollo {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedApollo{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedApollo) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
