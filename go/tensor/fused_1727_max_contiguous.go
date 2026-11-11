package tensor

import "fmt"

func Fused1727MaxContiguous(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("max_contiguous: nil operand")
	}
	return x.Max(0, false), nil
}
func Fused1727MaxContiguousName() string { return "max_contiguous" }
