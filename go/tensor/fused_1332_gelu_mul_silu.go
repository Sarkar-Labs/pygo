package tensor

import "fmt"

func Fused1332GeluMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.SiLU()
	return r, nil
}
func Fused1332GeluMulSiluName() string { return "gelu_mul_silu" }
