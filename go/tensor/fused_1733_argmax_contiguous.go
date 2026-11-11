package tensor

import "fmt"

func Fused1733ArgmaxContiguous(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("argmax_contiguous: nil operand")
	}
	return x.Argmax(0), nil
}
func Fused1733ArgmaxContiguousName() string { return "argmax_contiguous" }
