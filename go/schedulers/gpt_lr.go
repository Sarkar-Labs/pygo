package schedulers

import "math"

// GeneratedGptLr implements a deterministic scalar update corresponding to gpt lr.
type GeneratedGptLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedGptLr(rate float64) *GeneratedGptLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedGptLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedGptLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
