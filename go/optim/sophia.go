package optim

import "math"

// GeneratedSophia implements a deterministic scalar update corresponding to sophia.
type GeneratedSophia struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedSophia(rate float64) *GeneratedSophia {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedSophia{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedSophia) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
