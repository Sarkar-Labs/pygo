package optim

import "math"

// GeneratedAdahessian implements a deterministic scalar update corresponding to adahessian.
type GeneratedAdahessian struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedAdahessian(rate float64) *GeneratedAdahessian {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedAdahessian{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedAdahessian) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
