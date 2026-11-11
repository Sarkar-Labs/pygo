package tensor

import "fmt"

func Fused1456NegSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Abs()
	return r, nil
}
func Fused1456NegSubAbsName() string { return "neg_sub_abs" }
