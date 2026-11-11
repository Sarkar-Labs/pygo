package schedulers

import "math"

// GeneratedNoamLr implements a deterministic scalar update corresponding to noam lr.
type GeneratedNoamLr struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedNoamLr(rate float64) *GeneratedNoamLr {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedNoamLr{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedNoamLr) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
