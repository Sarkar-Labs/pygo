package optim

import "math"

// GeneratedLinearLr implements a deterministic scalar update corresponding to linear lr.
type GeneratedLinearLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedLinearLr(rate float64) *GeneratedLinearLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedLinearLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedLinearLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
