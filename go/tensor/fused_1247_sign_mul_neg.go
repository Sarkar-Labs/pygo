package tensor

import "fmt"

func Fused1247SignMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Neg()
	return r, nil
}
func Fused1247SignMulNegName() string { return "sign_mul_neg" }
