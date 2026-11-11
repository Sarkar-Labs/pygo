package tensor

import "fmt"

func Fused0608ExpDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_gelu: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.GELU()
	return r, nil
}
func Fused0608ExpDivGeluName() string { return "exp_div_gelu" }
