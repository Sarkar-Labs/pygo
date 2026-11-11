package optim

import "math"

// GeneratedSparseAdam implements a deterministic scalar update corresponding to sparse adam.
type GeneratedSparseAdam struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedSparseAdam(rate float64) *GeneratedSparseAdam {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedSparseAdam{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedSparseAdam) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
