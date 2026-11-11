package tensor

import "fmt"

func Fused0097ReluMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Sigmoid()
	return r, nil
}
func Fused0097ReluMulSigmoidName() string { return "relu_mul_sigmoid" }
