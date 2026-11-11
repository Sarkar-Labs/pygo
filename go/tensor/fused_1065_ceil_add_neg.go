package tensor

import "fmt"

func Fused1065CeilAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Neg()
	return r, nil
}
func Fused1065CeilAddNegName() string { return "ceil_add_neg" }
