package optim

import "math"

// GeneratedSchedulerBase implements a deterministic scalar update corresponding to scheduler base.
type GeneratedSchedulerBase struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedSchedulerBase(rate float64) *GeneratedSchedulerBase {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedSchedulerBase{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedSchedulerBase) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
