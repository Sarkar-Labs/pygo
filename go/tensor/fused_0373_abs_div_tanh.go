package tensor

import "fmt"

func Fused0373AbsDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Tanh()
	return r, nil
}
func Fused0373AbsDivTanhName() string { return "abs_div_tanh" }
