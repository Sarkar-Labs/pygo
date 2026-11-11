package tensor

import "fmt"

func Fused1715SumContiguous(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sum_contiguous: nil operand")
	}
	return x.Sum(0, false), nil
}
func Fused1715SumContiguousName() string { return "sum_contiguous" }
