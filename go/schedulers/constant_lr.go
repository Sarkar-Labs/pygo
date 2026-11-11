package schedulers

import "math"

// GeneratedConstantLr implements a deterministic scalar update corresponding to constant lr.
type GeneratedConstantLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedConstantLr(rate float64) *GeneratedConstantLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedConstantLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedConstantLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
