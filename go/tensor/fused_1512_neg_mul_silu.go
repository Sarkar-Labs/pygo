package tensor

import "fmt"

func Fused1512NegMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.SiLU()
	return r, nil
}
func Fused1512NegMulSiluName() string { return "neg_mul_silu" }
