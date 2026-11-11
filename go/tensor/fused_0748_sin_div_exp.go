package tensor

import "fmt"

func Fused0748SinDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Exp()
	return r, nil
}
func Fused0748SinDivExpName() string { return "sin_div_exp" }
