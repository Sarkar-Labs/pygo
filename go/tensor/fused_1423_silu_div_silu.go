package tensor

import "fmt"

func Fused1423SiluDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.SiLU()
	return r, nil
}
func Fused1423SiluDivSiluName() string { return "silu_div_silu" }
