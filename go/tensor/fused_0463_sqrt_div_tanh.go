package tensor

import "fmt"

func Fused0463SqrtDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Tanh()
	return r, nil
}
func Fused0463SqrtDivTanhName() string { return "sqrt_div_tanh" }
