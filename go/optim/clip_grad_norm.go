package optim

import "math"

// GeneratedClipGradNorm implements a deterministic scalar update corresponding to clip grad norm.
type GeneratedClipGradNorm struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedClipGradNorm(rate float64) *GeneratedClipGradNorm {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedClipGradNorm{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedClipGradNorm) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
