package schedulers

import "math"

// GeneratedSequentialLr implements a deterministic scalar update corresponding to sequential lr.
type GeneratedSequentialLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedSequentialLr(rate float64) *GeneratedSequentialLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedSequentialLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedSequentialLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
