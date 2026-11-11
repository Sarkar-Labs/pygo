package tensor

import "fmt"

func Fused0733SinDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Tanh()
	return r, nil
}
func Fused0733SinDivTanhName() string { return "sin_div_tanh" }
