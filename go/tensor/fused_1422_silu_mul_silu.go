package tensor

import "fmt"

func Fused1422SiluMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.SiLU()
	return r, nil
}
func Fused1422SiluMulSiluName() string { return "silu_mul_silu" }
