package tensor

import "fmt"

func Fused1093RoundDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Tanh()
	return r, nil
}
func Fused1093RoundDivTanhName() string { return "round_div_tanh" }
