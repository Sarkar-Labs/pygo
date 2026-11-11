package optim

import "math"

// GeneratedLion implements a deterministic scalar update corresponding to lion.
type GeneratedLion struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedLion(rate float64) *GeneratedLion {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedLion{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedLion) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
