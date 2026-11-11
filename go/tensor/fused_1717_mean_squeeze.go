package tensor

import "fmt"

func Fused1717MeanSqueeze(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("mean_squeeze: nil operand")
	}
	return x.Mean(0, false), nil
}
func Fused1717MeanSqueezeName() string { return "mean_squeeze" }
