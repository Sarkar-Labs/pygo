package tensor

import "fmt"

func Fused0792SinMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.SiLU()
	return r, nil
}
func Fused0792SinMulSiluName() string { return "sin_mul_silu" }
