package tensor

import "fmt"

func Fused1491NegSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Ceil()
	return r, nil
}
func Fused1491NegSubCeilName() string { return "neg_sub_ceil" }
