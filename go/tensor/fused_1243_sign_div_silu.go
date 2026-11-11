package tensor

import "fmt"

func Fused1243SignDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_silu: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.SiLU()
	return r, nil
}
func Fused1243SignDivSiluName() string { return "sign_div_silu" }
