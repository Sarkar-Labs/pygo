package tensor

import "fmt"

func Fused0527SqrtMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Neg()
	return r, nil
}
func Fused0527SqrtMulNegName() string { return "sqrt_mul_neg" }
