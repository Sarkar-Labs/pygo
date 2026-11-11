package tensor

import "fmt"

func Fused1438SiluDivLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_log10: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Log10()
	return r, nil
}
func Fused1438SiluDivLog10Name() string { return "silu_div_log10" }
