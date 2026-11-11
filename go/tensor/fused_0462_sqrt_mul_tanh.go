package tensor

import "fmt"

func Fused0462SqrtMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Tanh()
	return r, nil
}
func Fused0462SqrtMulTanhName() string { return "sqrt_mul_tanh" }
