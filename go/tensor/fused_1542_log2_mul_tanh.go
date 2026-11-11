package tensor

import "fmt"

func Fused1542Log2MulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Tanh()
	return r, nil
}
func Fused1542Log2MulTanhName() string { return "log2_mul_tanh" }
