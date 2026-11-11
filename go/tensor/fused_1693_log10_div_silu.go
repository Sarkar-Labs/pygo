package tensor

import "fmt"

func Fused1693Log10DivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Log10()
	r = r.SiLU()
	return r, nil
}
func Fused1693Log10DivSiluName() string { return "log10_div_silu" }
