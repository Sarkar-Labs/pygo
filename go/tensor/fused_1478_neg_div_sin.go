package tensor

import "fmt"

func Fused1478NegDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Sin()
	return r, nil
}
func Fused1478NegDivSinName() string { return "neg_div_sin" }
