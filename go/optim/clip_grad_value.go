package optim

import "math"

// GeneratedClipGradValue implements a deterministic scalar update corresponding to clip grad value.
type GeneratedClipGradValue struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedClipGradValue(rate float64) *GeneratedClipGradValue {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedClipGradValue{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedClipGradValue) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
