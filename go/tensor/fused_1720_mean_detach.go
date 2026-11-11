package tensor

import "fmt"

func Fused1720MeanDetach(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("mean_detach: nil operand")
	}
	return x.Mean(0, false), nil
}
func Fused1720MeanDetachName() string { return "mean_detach" }
