package tensor

import "fmt"

func Fused1483NegDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Cos()
	return r, nil
}
func Fused1483NegDivCosName() string { return "neg_div_cos" }
