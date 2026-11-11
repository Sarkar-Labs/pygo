package losses

import "math"

// GeneratedGaussianNll computes a stable scalar score for gaussian nll.
func GeneratedGaussianNll(prediction, target []float64) float64 {
	n := len(prediction)
	if len(target) < n {
		n = len(target)
	}
	if n == 0 {
		return 0
	}
	var total float64
	for i := 0; i < n; i++ {
		d := prediction[i] - target[i]
		total += d * d
	}
	return math.Sqrt(total / float64(n))
}

// GeneratedGaussianNllGradient returns the elementwise derivative of the squared error.
func GeneratedGaussianNllGradient(prediction, target []float64) []float64 {
	n := len(prediction)
	if len(target) < n {
		n = len(target)
	}
	out := make([]float64, n)
	if n == 0 {
		return out
	}
	for i := 0; i < n; i++ {
		out[i] = 2 * (prediction[i] - target[i]) / float64(n)
	}
	return out
}
