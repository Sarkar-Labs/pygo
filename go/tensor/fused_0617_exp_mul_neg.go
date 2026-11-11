package tensor

import "fmt"

func Fused0617ExpMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Neg()
	return r, nil
}
func Fused0617ExpMulNegName() string { return "exp_mul_neg" }
