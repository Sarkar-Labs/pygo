package tensor

import "fmt"

func Fused0182SigmoidMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.ReLU()
	return r, nil
}
func Fused0182SigmoidMulReluName() string { return "sigmoid_mul_relu" }
