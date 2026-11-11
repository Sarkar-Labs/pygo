package tensor

import "fmt"

func Fused0592ExpMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Ceil()
	return r, nil
}
func Fused0592ExpMulCeilName() string { return "exp_mul_ceil" }
