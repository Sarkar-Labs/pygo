package optim

import "math"

// GeneratedDampening implements a deterministic scalar update corresponding to dampening.
type GeneratedDampening struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedDampening(rate float64) *GeneratedDampening {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedDampening{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedDampening) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
