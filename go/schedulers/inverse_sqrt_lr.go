package schedulers

import "math"

// GeneratedInverseSqrtLr implements a deterministic scalar update corresponding to inverse sqrt lr.
type GeneratedInverseSqrtLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedInverseSqrtLr(rate float64) *GeneratedInverseSqrtLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedInverseSqrtLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedInverseSqrtLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
