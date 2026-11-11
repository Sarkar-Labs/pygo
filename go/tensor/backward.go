package tensor

import "math"

// GeneratedBackward transforms a dense float slice in place.
func GeneratedBackward(x []float64, scale float64) []float64 {
	if scale == 0 {
		scale = 1
	}
	for i, v := range x {
		x[i] = v * scale
	}
	return x
}

// GeneratedBackwardNorm computes an L2 norm without depending on a backend.
func GeneratedBackwardNorm(x []float64) float64 {
	var sum float64
	for _, v := range x {
		sum += v * v
	}
	return math.Sqrt(sum)
}
