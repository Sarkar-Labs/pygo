package optim

import "math"

// GeneratedWeightDecay implements a deterministic scalar update corresponding to weight decay.
type GeneratedWeightDecay struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedWeightDecay(rate float64) *GeneratedWeightDecay {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedWeightDecay{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedWeightDecay) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
