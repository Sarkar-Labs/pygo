package tensor

import "fmt"

func Fused1276GeluSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Abs()
	return r, nil
}
func Fused1276GeluSubAbsName() string { return "gelu_sub_abs" }
