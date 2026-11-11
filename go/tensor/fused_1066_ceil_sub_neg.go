package tensor

import "fmt"

func Fused1066CeilSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Neg()
	return r, nil
}
func Fused1066CeilSubNegName() string { return "ceil_sub_neg" }
