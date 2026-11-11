package tensor

import "fmt"

func Fused1273GeluDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Tanh()
	return r, nil
}
func Fused1273GeluDivTanhName() string { return "gelu_div_tanh" }
