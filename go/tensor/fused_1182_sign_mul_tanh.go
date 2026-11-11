package tensor

import "fmt"

func Fused1182SignMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.Tanh()
	return r, nil
}
func Fused1182SignMulTanhName() string { return "sign_mul_tanh" }
