package tensor

import "fmt"

func Fused0732SinMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Tanh()
	return r, nil
}
func Fused0732SinMulTanhName() string { return "sin_mul_tanh" }
