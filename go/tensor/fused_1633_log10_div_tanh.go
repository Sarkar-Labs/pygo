package tensor

import "fmt"

func Fused1633Log10DivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.Tanh()
	return r, nil
}
func Fused1633Log10DivTanhName() string { return "log10_div_tanh" }
