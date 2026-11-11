package optim

import "math"

// GeneratedCame implements a deterministic scalar update corresponding to came.
type GeneratedCame struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedCame(rate float64) *GeneratedCame {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedCame{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedCame) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
