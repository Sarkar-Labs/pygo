package tensor

import "fmt"

func Fused0378AbsDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Abs()
	return r, nil
}
func Fused0378AbsDivAbsName() string { return "abs_div_abs" }
