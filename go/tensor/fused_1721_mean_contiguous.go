package tensor

import "fmt"

func Fused1721MeanContiguous(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("mean_contiguous: nil operand")
	}
	return x.Mean(0, false), nil
}
func Fused1721MeanContiguousName() string { return "mean_contiguous" }
