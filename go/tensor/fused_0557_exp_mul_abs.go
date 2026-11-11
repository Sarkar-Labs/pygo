package tensor

import "fmt"

func Fused0557ExpMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Abs()
	return r, nil
}
func Fused0557ExpMulAbsName() string { return "exp_mul_abs" }
