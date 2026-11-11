package optim

import "math"

// GeneratedParamGroup implements a deterministic scalar update corresponding to param group.
type GeneratedParamGroup struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedParamGroup(rate float64) *GeneratedParamGroup {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedParamGroup{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedParamGroup) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
