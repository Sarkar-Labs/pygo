package tensor

import "fmt"

func Fused0436AbsSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Neg()
	return r, nil
}
func Fused0436AbsSubNegName() string { return "abs_sub_neg" }
