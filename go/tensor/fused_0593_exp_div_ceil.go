package tensor

import "fmt"

func Fused0593ExpDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Ceil()
	return r, nil
}
func Fused0593ExpDivCeilName() string { return "exp_div_ceil" }
