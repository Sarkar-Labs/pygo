package tensor

import "fmt"

func Fused1288GeluDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Exp()
	return r, nil
}
func Fused1288GeluDivExpName() string { return "gelu_div_exp" }
