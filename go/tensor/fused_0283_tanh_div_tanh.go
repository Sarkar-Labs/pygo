package tensor

import "fmt"

func Fused0283TanhDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Tanh()
	return r, nil
}
func Fused0283TanhDivTanhName() string { return "tanh_div_tanh" }
