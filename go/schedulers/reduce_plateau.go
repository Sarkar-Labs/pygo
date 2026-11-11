package schedulers

import "math"

// GeneratedReducePlateau implements a deterministic scalar update corresponding to reduce plateau.
type GeneratedReducePlateau struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedReducePlateau(rate float64) *GeneratedReducePlateau {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedReducePlateau{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedReducePlateau) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
