package schedulers

import "math"

// GeneratedTransformerLr implements a deterministic scalar update corresponding to transformer lr.
type GeneratedTransformerLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedTransformerLr(rate float64) *GeneratedTransformerLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedTransformerLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedTransformerLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
