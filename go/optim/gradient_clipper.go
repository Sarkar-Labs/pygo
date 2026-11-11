package optim

import "math"

// GeneratedGradientClipper implements a deterministic scalar update corresponding to gradient clipper.
type GeneratedGradientClipper struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedGradientClipper(rate float64) *GeneratedGradientClipper {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedGradientClipper{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedGradientClipper) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
