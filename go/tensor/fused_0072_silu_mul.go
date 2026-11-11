package tensor

import "fmt"

func Fused0072SiluMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	return r, nil
}
func Fused0072SiluMulName() string { return "silu_mul" }
