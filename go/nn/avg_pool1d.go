package nn

import "math"

// GeneratedAvgPool1d applies a numerically stable activation primitive.
func GeneratedAvgPool1d(x []float64) []float64 {
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

// GeneratedAvgPool1dInPlace updates a slice without allocating another tensor.
func GeneratedAvgPool1dInPlace(x []float64) {
	for i, v := range x {
		if v > 20 {
			x[i] = v
		} else {
			x[i] = math.Log1p(math.Exp(v))
		}
	}
}
