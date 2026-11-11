package tensor

import "fmt"

func Fused1063CeilDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.SiLU()
	return r, nil
}
func Fused1063CeilDivSiluName() string { return "ceil_div_silu" }
