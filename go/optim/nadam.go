package optim

import "math"

// GeneratedNadam implements a deterministic scalar update corresponding to nadam.
type GeneratedNadam struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedNadam(rate float64) *GeneratedNadam {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedNadam{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedNadam) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
