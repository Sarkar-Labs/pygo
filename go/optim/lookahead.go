package optim

import "math"

// GeneratedLookahead implements a deterministic scalar update corresponding to lookahead.
type GeneratedLookahead struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedLookahead(rate float64) *GeneratedLookahead {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedLookahead{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedLookahead) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
