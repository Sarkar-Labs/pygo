package tensor

import "fmt"

func Fused0372AbsMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Tanh()
	return r, nil
}
func Fused0372AbsMulTanhName() string { return "abs_mul_tanh" }
