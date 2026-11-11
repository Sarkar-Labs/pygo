package schedulers

import "math"

// GeneratedExponentialLr implements a deterministic scalar update corresponding to exponential lr.
type GeneratedExponentialLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedExponentialLr(rate float64) *GeneratedExponentialLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedExponentialLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedExponentialLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
