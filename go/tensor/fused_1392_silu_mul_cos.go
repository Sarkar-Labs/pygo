package tensor

import "fmt"

func Fused1392SiluMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Cos()
	return r, nil
}
func Fused1392SiluMulCosName() string { return "silu_mul_cos" }
