package schedulers

import "math"

// GeneratedWarmupLr implements a deterministic scalar update corresponding to warmup lr.
type GeneratedWarmupLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedWarmupLr(rate float64) *GeneratedWarmupLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedWarmupLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedWarmupLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
