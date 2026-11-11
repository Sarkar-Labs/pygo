package tensor

import "fmt"

func Fused0552ExpMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Tanh()
	return r, nil
}
func Fused0552ExpMulTanhName() string { return "exp_mul_tanh" }
