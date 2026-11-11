package tensor

import "fmt"

func Fused1062CeilMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_silu: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.SiLU()
	return r, nil
}
func Fused1062CeilMulSiluName() string { return "ceil_mul_silu" }
