package tensor

import "fmt"

func Fused1492NegMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Ceil()
	return r, nil
}
func Fused1492NegMulCeilName() string { return "neg_mul_ceil" }
