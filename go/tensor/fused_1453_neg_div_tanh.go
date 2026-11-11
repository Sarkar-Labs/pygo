package tensor

import "fmt"

func Fused1453NegDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Tanh()
	return r, nil
}
func Fused1453NegDivTanhName() string { return "neg_div_tanh" }
