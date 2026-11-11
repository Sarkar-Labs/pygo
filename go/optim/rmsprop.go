package optim

import "math"

// GeneratedRmsprop implements a deterministic scalar update corresponding to rmsprop.
type GeneratedRmsprop struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedRmsprop(rate float64) *GeneratedRmsprop {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedRmsprop{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedRmsprop) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
