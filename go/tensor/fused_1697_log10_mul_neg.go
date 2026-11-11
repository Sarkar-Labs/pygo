package tensor

import "fmt"

func Fused1697Log10MulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Neg()
	return r, nil
}
func Fused1697Log10MulNegName() string { return "log10_mul_neg" }
