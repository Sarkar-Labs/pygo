package tensor

import "fmt"

func Fused1515NegAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	r = r.Neg()
	return r, nil
}
func Fused1515NegAddNegName() string { return "neg_add_neg" }
