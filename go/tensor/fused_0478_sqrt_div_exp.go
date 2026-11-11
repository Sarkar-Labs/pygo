package tensor

import "fmt"

func Fused0478SqrtDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Exp()
	return r, nil
}
func Fused0478SqrtDivExpName() string { return "sqrt_div_exp" }
