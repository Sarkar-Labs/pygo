package optim

import "math"

// GeneratedLamb implements a deterministic scalar update corresponding to lamb.
type GeneratedLamb struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedLamb(rate float64) *GeneratedLamb {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedLamb{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedLamb) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
