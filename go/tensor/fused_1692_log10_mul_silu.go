package tensor

import "fmt"

func Fused1692Log10MulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.SiLU()
	return r, nil
}
func Fused1692Log10MulSiluName() string { return "log10_mul_silu" }
