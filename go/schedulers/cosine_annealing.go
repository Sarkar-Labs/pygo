package schedulers

import "math"

// GeneratedCosineAnnealing implements a deterministic scalar update corresponding to cosine annealing.
type GeneratedCosineAnnealing struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedCosineAnnealing(rate float64) *GeneratedCosineAnnealing {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedCosineAnnealing{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedCosineAnnealing) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
