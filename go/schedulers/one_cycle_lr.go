package schedulers

import "math"

// GeneratedOneCycleLr implements a deterministic scalar update corresponding to one cycle lr.
type GeneratedOneCycleLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedOneCycleLr(rate float64) *GeneratedOneCycleLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedOneCycleLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedOneCycleLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
