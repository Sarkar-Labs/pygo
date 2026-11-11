package tensor

import "fmt"

func Fused1467NegMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Exp()
	return r, nil
}
func Fused1467NegMulExpName() string { return "neg_mul_exp" }
