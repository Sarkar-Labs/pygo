package tensor

import "fmt"

func Fused0882CosMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.SiLU()
	return r, nil
}
func Fused0882CosMulSiluName() string { return "cos_mul_silu" }
