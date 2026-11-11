package tensor

import "fmt"

func Fused0577ExpMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Sin()
	return r, nil
}
func Fused0577ExpMulSinName() string { return "exp_mul_sin" }
