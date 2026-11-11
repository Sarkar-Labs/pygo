package tensor

import "fmt"

func Fused1718MeanUnsqueeze(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("mean_unsqueeze: nil operand")
	}
	return x.Mean(0, false), nil
}
func Fused1718MeanUnsqueezeName() string { return "mean_unsqueeze" }
