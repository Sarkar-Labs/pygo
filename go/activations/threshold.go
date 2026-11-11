package activations

import "math"

// GeneratedThreshold applies a numerically stable activation primitive.
func GeneratedThreshold(x []float64) []float64 {
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

// GeneratedThresholdInPlace updates a slice without allocating another tensor.
func GeneratedThresholdInPlace(x []float64) {
	for i, v := range x {
		if v > 20 {
			x[i] = v
		} else {
			x[i] = math.Log1p(math.Exp(v))
		}
	}
}
