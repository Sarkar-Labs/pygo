package losses

import "math"

// GeneratedSoftMargin computes a stable scalar score for soft margin.
func GeneratedSoftMargin(prediction, target []float64) float64 {
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

// GeneratedSoftMarginGradient returns the elementwise derivative of the squared error.
func GeneratedSoftMarginGradient(prediction, target []float64) []float64 {
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
