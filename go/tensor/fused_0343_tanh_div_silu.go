package tensor

import "fmt"

func Fused0343TanhDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.SiLU()
	return r, nil
}
func Fused0343TanhDivSiluName() string { return "tanh_div_silu" }
