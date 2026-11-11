package tensor

import "fmt"

func Fused0582ExpMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.Cos()
	return r, nil
}
func Fused0582ExpMulCosName() string { return "exp_mul_cos" }
