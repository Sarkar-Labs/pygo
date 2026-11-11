package optim

import "math"

// GeneratedMechanic implements a deterministic scalar update corresponding to mechanic.
type GeneratedMechanic struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedMechanic(rate float64) *GeneratedMechanic {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedMechanic{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedMechanic) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
