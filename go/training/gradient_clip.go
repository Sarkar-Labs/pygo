package training

import (
	"github.com/pygo/go/nn"
	"math"
)

type GradientClipper struct {
	MaxNorm float64
	history []float64
}

func NewGradClipper(maxNorm float64) *GradientClipper { return &GradientClipper{MaxNorm: maxNorm} }
func (gc *GradientClipper) Clip(params []*nn.Parameter) float64 {
	total := 0.0
	for _, p := range params {
		if g := p.Grad(); g != nil {
			n := g.Pow(2).Sum(-1, false).Sqrt().Item()
			total += n * n
		}
	}
	norm := math.Sqrt(total)
	if norm > gc.MaxNorm {
		coef := gc.MaxNorm / (norm + 1e-6)
		for _, p := range params {
			if g := p.Grad(); g != nil {
				_ = g.MulScalar(coef)
			}
		}
	}
	gc.history = append(gc.history, norm)
	if len(gc.history) > 100 {
		gc.history = gc.history[1:]
	}
	return norm
}
