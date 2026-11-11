package optim

import "math"

// GeneratedSam implements a deterministic scalar update corresponding to sam.
type GeneratedSam struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedSam(rate float64) *GeneratedSam {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedSam{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedSam) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
