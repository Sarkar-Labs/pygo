package schedulers

import "math"

// GeneratedBertLr implements a deterministic scalar update corresponding to bert lr.
type GeneratedBertLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedBertLr(rate float64) *GeneratedBertLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedBertLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedBertLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
