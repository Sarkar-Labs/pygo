package optim

import "math"

// GeneratedMuon implements a deterministic scalar update corresponding to muon.
type GeneratedMuon struct {
	Rate      float64
	StepCount int64
}

func NewGeneratedMuon(rate float64) *GeneratedMuon {
	if rate <= 0 {
		rate = 1e-3
	}
	return &GeneratedMuon{Rate: rate}
}

// Step applies one numerically stable update to a scalar parameter.
func (o *GeneratedMuon) Step(value, gradient float64) float64 {
	o.StepCount++
	denom := math.Max(1, math.Abs(gradient))
	return value - o.Rate*gradient/denom
}
