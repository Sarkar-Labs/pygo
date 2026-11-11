package optim

import "math"

// GeneratedAmp implements a deterministic scalar update corresponding to amp.
type GeneratedAmp struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedAmp(rate float64) *GeneratedAmp {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedAmp{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedAmp) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
