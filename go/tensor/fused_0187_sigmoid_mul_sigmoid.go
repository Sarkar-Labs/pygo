package tensor

import "fmt"

func Fused0187SigmoidMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Sigmoid()
	return r, nil
}
func Fused0187SigmoidMulSigmoidName() string { return "sigmoid_mul_sigmoid" }
