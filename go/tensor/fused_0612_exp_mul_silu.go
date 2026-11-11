package tensor

import "fmt"

func Fused0612ExpMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Exp()
	r = r.SiLU()
	return r, nil
}
func Fused0612ExpMulSiluName() string { return "exp_mul_silu" }
