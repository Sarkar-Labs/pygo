package tensor

import "fmt"

func Fused1333GeluDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.SiLU()
	return r, nil
}
func Fused1333GeluDivSiluName() string { return "gelu_div_silu" }
