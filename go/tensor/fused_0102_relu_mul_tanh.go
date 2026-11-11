package tensor

import "fmt"

func Fused0102ReluMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Tanh()
	return r, nil
}
func Fused0102ReluMulTanhName() string { return "relu_mul_tanh" }
