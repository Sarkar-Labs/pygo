package tensor

import "fmt"

func Fused1362SiluMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.SiLU()
	r = r.Tanh()
	return r, nil
}
func Fused1362SiluMulTanhName() string { return "silu_mul_tanh" }
