package schedulers

import "math"

// GeneratedDiffusionLr implements a deterministic scalar update corresponding to diffusion lr.
type GeneratedDiffusionLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedDiffusionLr(rate float64) *GeneratedDiffusionLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedDiffusionLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedDiffusionLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
