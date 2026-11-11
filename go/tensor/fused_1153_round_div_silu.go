package tensor

import "fmt"

func Fused1153RoundDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.SiLU()
	return r, nil
}
func Fused1153RoundDivSiluName() string { return "round_div_silu" }
