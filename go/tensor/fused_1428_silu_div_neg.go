package tensor

import "fmt"

func Fused1428SiluDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Neg()
	return r, nil
}
func Fused1428SiluDivNegName() string { return "silu_div_neg" }
