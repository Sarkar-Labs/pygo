package tensor

import "fmt"

func Fused0433AbsDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.SiLU()
	return r, nil
}
func Fused0433AbsDivSiluName() string { return "abs_div_silu" }
