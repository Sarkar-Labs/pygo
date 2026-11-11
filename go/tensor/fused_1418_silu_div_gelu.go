package tensor

import "fmt"

func Fused1418SiluDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.GELU()
	return r, nil
}
func Fused1418SiluDivGeluName() string { return "silu_div_gelu" }
