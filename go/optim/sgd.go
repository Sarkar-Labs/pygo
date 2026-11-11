package optim

import "math"

// GeneratedSgd implements a deterministic scalar update corresponding to sgd.
type GeneratedSgd struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedSgd(rate float64) *GeneratedSgd {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedSgd{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedSgd) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
