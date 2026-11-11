package tensor

import "fmt"

func Fused0613ExpDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.SiLU()
	return r, nil
}
func Fused0613ExpDivSiluName() string { return "exp_div_silu" }
