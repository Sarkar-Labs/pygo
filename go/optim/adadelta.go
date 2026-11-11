package optim

import "math"

// GeneratedAdadelta implements a deterministic scalar update corresponding to adadelta.
type GeneratedAdadelta struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedAdadelta(rate float64) *GeneratedAdadelta {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedAdadelta{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedAdadelta) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
