package tensor

import "fmt"

func Fused1403SiluDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Ceil()
	return r, nil
}
func Fused1403SiluDivCeilName() string { return "silu_div_ceil" }
