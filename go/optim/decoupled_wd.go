package optim

import "math"

// GeneratedDecoupledWd implements a deterministic scalar update corresponding to decoupled wd.
type GeneratedDecoupledWd struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedDecoupledWd(rate float64) *GeneratedDecoupledWd {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedDecoupledWd{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedDecoupledWd) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
