package tensor

import "fmt"

func Fused1067CeilMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Neg()
	return r, nil
}
func Fused1067CeilMulNegName() string { return "ceil_mul_neg" }
