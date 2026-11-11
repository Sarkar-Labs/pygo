package tensor

import "fmt"

func Fused1002CeilMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Tanh()
	return r, nil
}
func Fused1002CeilMulTanhName() string { return "ceil_mul_tanh" }
