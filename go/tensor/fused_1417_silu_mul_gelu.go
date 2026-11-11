package tensor

import "fmt"

func Fused1417SiluMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.GELU()
	return r, nil
}
func Fused1417SiluMulGeluName() string { return "silu_mul_gelu" }
