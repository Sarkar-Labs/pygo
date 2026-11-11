package optim

import "math"

// GeneratedAsam implements a deterministic scalar update corresponding to asam.
type GeneratedAsam struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedAsam(rate float64) *GeneratedAsam {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedAsam{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedAsam) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
