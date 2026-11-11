package optim

import "math"

// GeneratedAdan implements a deterministic scalar update corresponding to adan.
type GeneratedAdan struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedAdan(rate float64) *GeneratedAdan {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedAdan{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedAdan) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
