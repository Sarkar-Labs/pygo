package schedulers

import "math"

// GeneratedStepLr implements a deterministic scalar update corresponding to step lr.
type GeneratedStepLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedStepLr(rate float64) *GeneratedStepLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedStepLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedStepLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
