package tensor

import "fmt"

func Fused0797SinMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Neg()
	return r, nil
}
func Fused0797SinMulNegName() string { return "sin_mul_neg" }
