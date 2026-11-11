package tensor

import "fmt"

func Fused0822CosMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Tanh()
	return r, nil
}
func Fused0822CosMulTanhName() string { return "cos_mul_tanh" }
