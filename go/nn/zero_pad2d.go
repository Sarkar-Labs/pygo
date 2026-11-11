package nn

import "math"

// GeneratedZeroPad2d applies a numerically stable activation primitive.
func GeneratedZeroPad2d(x []float64) []float64 {
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

// GeneratedZeroPad2dInPlace updates a slice without allocating another tensor.
func GeneratedZeroPad2dInPlace(x []float64) {
	for i, v := range x {
		if v > 20 {
			x[i] = v
		} else {
			x[i] = math.Log1p(math.Exp(v))
		}
	}
}
