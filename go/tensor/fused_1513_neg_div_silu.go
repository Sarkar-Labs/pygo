package tensor

import "fmt"

func Fused1513NegDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.SiLU()
	return r, nil
}
func Fused1513NegDivSiluName() string { return "neg_div_silu" }
