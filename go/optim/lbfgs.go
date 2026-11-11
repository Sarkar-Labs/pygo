package optim

import "math"

// GeneratedLbfgs implements a deterministic scalar update corresponding to lbfgs.
type GeneratedLbfgs struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedLbfgs(rate float64) *GeneratedLbfgs {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedLbfgs{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedLbfgs) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
