package optim

import "math"

// GeneratedChainedScheduler implements a deterministic scalar update corresponding to chained scheduler.
type GeneratedChainedScheduler struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedChainedScheduler(rate float64) *GeneratedChainedScheduler {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedChainedScheduler{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedChainedScheduler) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
