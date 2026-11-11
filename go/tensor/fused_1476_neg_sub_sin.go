package tensor

import "fmt"

func Fused1476NegSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Sin()
	return r, nil
}
func Fused1476NegSubSinName() string { return "neg_sub_sin" }
