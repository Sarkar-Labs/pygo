package tensor

import "fmt"

func Fused1272GeluMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Tanh()
	return r, nil
}
func Fused1272GeluMulTanhName() string { return "gelu_mul_tanh" }
