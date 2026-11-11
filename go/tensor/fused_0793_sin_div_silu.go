package tensor

import "fmt"

func Fused0793SinDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.SiLU()
	return r, nil
}
func Fused0793SinDivSiluName() string { return "sin_div_silu" }
