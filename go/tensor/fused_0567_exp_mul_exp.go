package tensor

import "fmt"

func Fused0567ExpMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Exp()
	return r, nil
}
func Fused0567ExpMulExpName() string { return "exp_mul_exp" }
