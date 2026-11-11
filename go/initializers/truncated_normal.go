package initializers

import "math"

// GeneratedTruncatedNormal transforms a dense float slice in place.
func GeneratedTruncatedNormal(x []float64, scale float64) []float64 {
	if scale == 0 {
		scale = 1
	}
	for i, v := range x {
		x[i] = v * scale
	}
	return x
}

// GeneratedTruncatedNormalNorm computes an L2 norm without depending on a backend.
func GeneratedTruncatedNormalNorm(x []float64) float64 {
	var sum float64
	for _, v := range x {
		sum += v * v
	}
	return math.Sqrt(sum)
}
