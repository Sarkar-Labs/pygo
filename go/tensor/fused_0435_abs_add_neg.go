package tensor

import "fmt"

func Fused0435AbsAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Neg()
	return r, nil
}
func Fused0435AbsAddNegName() string { return "abs_add_neg" }
