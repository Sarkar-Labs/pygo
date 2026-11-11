package optim

import "math"

// GeneratedRadam implements a deterministic scalar update corresponding to radam.
type GeneratedRadam struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedRadam(rate float64) *GeneratedRadam {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedRadam{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedRadam) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
