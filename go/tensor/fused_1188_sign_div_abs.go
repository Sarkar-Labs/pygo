package tensor

import "fmt"

func Fused1188SignDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Abs()
	return r, nil
}
func Fused1188SignDivAbsName() string { return "sign_div_abs" }
