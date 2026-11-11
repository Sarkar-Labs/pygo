package tensor

import "fmt"

func Fused1739MinContiguous(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("min_contiguous: nil operand")
	}
	return x, nil
}
func Fused1739MinContiguousName() string { return "min_contiguous" }
