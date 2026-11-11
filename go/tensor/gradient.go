package tensor

// ClipGradNorm clips gradients of params to max_norm; returns actual norm
func ClipGradNorm(params []*Tensor, maxNorm float64) float64 {
	total := 0.0
	for _, p := range params {
		if g := p.Grad(); g != nil {
			n := g.Pow(2).Sum(-1, false).Sqrt().Item()
			total += n * n
		}
	}
	norm := total
	if norm == 0 {
		return 0
	}
	coef := maxNorm / (norm + 1e-6)
	if coef < 1.0 {
		for _, p := range params {
			if g := p.Grad(); g != nil {
				_ = g.MulScalar(coef)
			}
		}
	}
	return norm
}

// GradNorm computes the L2 norm of all gradients
func GradNorm(params []*Tensor) float64 {
	total := 0.0
	for _, p := range params {
		if g := p.Grad(); g != nil {
			n := g.Pow(2).Sum(-1, false).Sqrt().Item()
			total += n * n
		}
	}
	return total
}
