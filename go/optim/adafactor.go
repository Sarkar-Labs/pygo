package optim

import "math"

// GeneratedAdafactor implements a deterministic scalar update corresponding to adafactor.
type GeneratedAdafactor struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedAdafactor(rate float64) *GeneratedAdafactor {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedAdafactor{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedAdafactor) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
