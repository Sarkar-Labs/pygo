package optim

import "math"

// GeneratedLossScaler implements a deterministic scalar update corresponding to loss scaler.
type GeneratedLossScaler struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedLossScaler(rate float64) *GeneratedLossScaler {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedLossScaler{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedLossScaler) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
