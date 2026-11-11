package tensor

import "fmt"

func Fused0426AbsSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.GELU()
	return r, nil
}
func Fused0426AbsSubGeluName() string { return "abs_sub_gelu" }
