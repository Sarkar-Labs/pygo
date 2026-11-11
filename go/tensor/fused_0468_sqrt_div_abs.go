package tensor

import "fmt"

func Fused0468SqrtDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Abs()
	return r, nil
}
func Fused0468SqrtDivAbsName() string { return "sqrt_div_abs" }
