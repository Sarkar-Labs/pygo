package tensor

import "fmt"

func Fused1337GeluMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Neg()
	return r, nil
}
func Fused1337GeluMulNegName() string { return "gelu_mul_neg" }
