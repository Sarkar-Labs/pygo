package tensor

import "fmt"

func Fused0282TanhMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Tanh()
	return r, nil
}
func Fused0282TanhMulTanhName() string { return "tanh_mul_tanh" }
