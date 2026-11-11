package optim

import "math"

// GeneratedRprop implements a deterministic scalar update corresponding to rprop.
type GeneratedRprop struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedRprop(rate float64) *GeneratedRprop {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedRprop{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedRprop) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
