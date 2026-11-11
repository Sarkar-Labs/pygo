package nn

import "math"

// GeneratedDropout2d applies a numerically stable activation primitive.
func GeneratedDropout2d(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		if v > 20 {
			out[i] = v
		} else if v < -20 {
			out[i] = 0
		} else {
			out[i] = math.Log1p(math.Exp(v))
		}
	}
	return out
}

// GeneratedDropout2dInPlace updates a slice without allocating another tensor.
func GeneratedDropout2dInPlace(x []float64) {
	for i, v := range x {
		if v > 20 {
			x[i] = v
		} else {
			x[i] = math.Log1p(math.Exp(v))
		}
	}
}
