package optim

import "math"

// GeneratedGradientScaler implements a deterministic scalar update corresponding to gradient scaler.
type GeneratedGradientScaler struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedGradientScaler(rate float64) *GeneratedGradientScaler {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedGradientScaler{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedGradientScaler) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
