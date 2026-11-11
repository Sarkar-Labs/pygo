package metrics

import "github.com/pygo/go/tensor"

// MetricFn is a scalar metric over model predictions and targets.
type MetricFn func(predictions, targets *tensor.Tensor) float64
