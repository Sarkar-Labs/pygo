package tensor

import "fmt"

func Fused1388SiluDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Sin()
	return r, nil
}
func Fused1388SiluDivSinName() string { return "silu_div_sin" }
