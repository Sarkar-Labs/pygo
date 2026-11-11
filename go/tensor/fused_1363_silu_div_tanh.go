package tensor

import "fmt"

func Fused1363SiluDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Tanh()
	return r, nil
}
func Fused1363SiluDivTanhName() string { return "silu_div_tanh" }
