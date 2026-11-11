package tensor

import "fmt"

func Fused0558ExpDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Abs()
	return r, nil
}
func Fused0558ExpDivAbsName() string { return "exp_div_abs" }
