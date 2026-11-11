package tensor

import "fmt"

func Fused1372SiluMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Sqrt()
	return r, nil
}
func Fused1372SiluMulSqrtName() string { return "silu_mul_sqrt" }
