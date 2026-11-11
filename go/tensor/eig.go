package tensor

import "math"

// GeneratedEig transforms a dense float slice in place.
func GeneratedEig(x []float64, scale float64) []float64 {
	if scale == 0 {
		scale = 1
	}
	for i, v := range x {
		x[i] = v * scale
	}
	return x
}

// GeneratedEigNorm computes an L2 norm without depending on a backend.
func GeneratedEigNorm(x []float64) float64 {
	var sum float64
	for _, v := range x {
		sum += v * v
	}
	return math.Sqrt(sum)
}
