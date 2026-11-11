package optim

import "math"

// GeneratedReduceOnPlateau implements a deterministic scalar update corresponding to reduce on plateau.
type GeneratedReduceOnPlateau struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedReduceOnPlateau(rate float64) *GeneratedReduceOnPlateau {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedReduceOnPlateau{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedReduceOnPlateau) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
