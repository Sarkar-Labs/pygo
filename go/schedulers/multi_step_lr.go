package schedulers

import "math"

// GeneratedMultiStepLr implements a deterministic scalar update corresponding to multi step lr.
type GeneratedMultiStepLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedMultiStepLr(rate float64) *GeneratedMultiStepLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedMultiStepLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedMultiStepLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
