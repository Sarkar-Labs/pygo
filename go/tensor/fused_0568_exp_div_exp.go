package tensor

import "fmt"

func Fused0568ExpDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Exp()
	return r, nil
}
func Fused0568ExpDivExpName() string { return "exp_div_exp" }
