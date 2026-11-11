package tensor

import "fmt"

func Fused0523SqrtDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.SiLU()
	return r, nil
}
func Fused0523SqrtDivSiluName() string { return "sqrt_div_silu" }
