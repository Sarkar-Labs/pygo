package tensor

import "fmt"

func Fused1461NegSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Sqrt()
	return r, nil
}
func Fused1461NegSubSqrtName() string { return "neg_sub_sqrt" }
