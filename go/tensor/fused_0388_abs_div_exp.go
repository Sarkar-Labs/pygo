package tensor

import "fmt"

func Fused0388AbsDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Exp()
	return r, nil
}
func Fused0388AbsDivExpName() string { return "abs_div_exp" }
