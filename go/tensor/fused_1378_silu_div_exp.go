package tensor

import "fmt"

func Fused1378SiluDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Exp()
	return r, nil
}
func Fused1378SiluDivExpName() string { return "silu_div_exp" }
