package optim

import "math"

// GeneratedLambdaLr implements a deterministic scalar update corresponding to lambda lr.
type GeneratedLambdaLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedLambdaLr(rate float64) *GeneratedLambdaLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedLambdaLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedLambdaLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
