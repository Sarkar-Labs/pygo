package tensor

import "fmt"

func Fused0883CosDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.SiLU()
	return r, nil
}
func Fused0883CosDivSiluName() string { return "cos_div_silu" }
