package tensor

import "fmt"

func Fused0607ExpMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.GELU()
	return r, nil
}
func Fused0607ExpMulGeluName() string { return "exp_mul_gelu" }
