package tensor

import "fmt"

func Fused1516NegSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Neg()
	return r, nil
}
func Fused1516NegSubNegName() string { return "neg_sub_neg" }
