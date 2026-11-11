package tensor

import "fmt"

func Fused0562ExpMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Sqrt()
	return r, nil
}
func Fused0562ExpMulSqrtName() string { return "exp_mul_sqrt" }
