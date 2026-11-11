package tensor

import "fmt"

func Fused0092ReluMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.ReLU()
	return r, nil
}
func Fused0092ReluMulReluName() string { return "relu_mul_relu" }
