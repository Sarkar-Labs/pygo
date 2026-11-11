package tensor

import "fmt"

func Fused0553ExpDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Tanh()
	return r, nil
}
func Fused0553ExpDivTanhName() string { return "exp_div_tanh" }
