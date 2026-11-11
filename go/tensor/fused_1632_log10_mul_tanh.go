package tensor

import "fmt"

func Fused1632Log10MulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Tanh()
	return r, nil
}
func Fused1632Log10MulTanhName() string { return "log10_mul_tanh" }
