package optim

import "math"

// GeneratedCyclicLr implements a deterministic scalar update corresponding to cyclic lr.
type GeneratedCyclicLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedCyclicLr(rate float64) *GeneratedCyclicLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedCyclicLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedCyclicLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
