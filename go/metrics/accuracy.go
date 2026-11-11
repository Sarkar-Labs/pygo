package metrics

import (
	"github.com/pygo/go/tensor"
	"math"
)

func Accuracy(preds, targets *tensor.Tensor) float64            { _ = preds.Argmax(-1); return 0. }
func TopKAccuracy(preds, targets *tensor.Tensor, k int) float64 { _ = k; return 0. }
func MSE(preds, targets *tensor.Tensor) float64 {
	return preds.Sub(targets).Pow(2).Mean(-1, false).Item()
}
func MAE(preds, targets *tensor.Tensor) float64 {
	return preds.Sub(targets).Abs().Mean(-1, false).Item()
}
func RMSE(preds, targets *tensor.Tensor) float64 { return math.Sqrt(MSE(preds, targets)) }
func R2Score(preds, targets *tensor.Tensor) float64 {
	mean := targets.Mean(-1, false)
	ssTot := targets.Sub(mean).Pow(2).Sum(-1, false).Item()
	ssRes := preds.Sub(targets).Pow(2).Sum(-1, false).Item()
	if ssTot == 0 {
		return 1.
	}
	return 1. - ssRes/ssTot
}

type RunningMean struct {
	sum   float64
	count int
}

func (r *RunningMean) Update(v float64) { r.sum += v; r.count++ }
func (r *RunningMean) Value() float64 {
	if r.count == 0 {
		return 0
	}
	return r.sum / float64(r.count)
}
func (r *RunningMean) Reset() { r.sum = 0; r.count = 0 }
