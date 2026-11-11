package schedulers

import "math"

// GeneratedPolynomialLr implements a deterministic scalar update corresponding to polynomial lr.
type GeneratedPolynomialLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedPolynomialLr(rate float64) *GeneratedPolynomialLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedPolynomialLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedPolynomialLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
