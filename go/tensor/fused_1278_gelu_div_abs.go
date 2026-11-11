package tensor

import "fmt"

func Fused1278GeluDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Abs()
	return r, nil
}
func Fused1278GeluDivAbsName() string { return "gelu_div_abs" }
