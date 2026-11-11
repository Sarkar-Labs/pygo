package tensor

import "fmt"

func Fused1003CeilDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Tanh()
	return r, nil
}
func Fused1003CeilDivTanhName() string { return "ceil_div_tanh" }
