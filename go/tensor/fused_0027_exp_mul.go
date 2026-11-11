package tensor

import "fmt"

func Fused0027ExpMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	return r, nil
}
func Fused0027ExpMulName() string { return "exp_mul" }
