package tensor

import "fmt"

func Fused1517NegMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Neg()
	return r, nil
}
func Fused1517NegMulNegName() string { return "neg_mul_neg" }
