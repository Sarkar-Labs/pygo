package schedulers

import "math"

// GeneratedMultiplicativeLr implements a deterministic scalar update corresponding to multiplicative lr.
type GeneratedMultiplicativeLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedMultiplicativeLr(rate float64) *GeneratedMultiplicativeLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedMultiplicativeLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedMultiplicativeLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
