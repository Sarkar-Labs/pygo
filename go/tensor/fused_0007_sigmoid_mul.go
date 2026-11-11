package tensor

import "fmt"

func Fused0007SigmoidMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	return r, nil
}
func Fused0007SigmoidMulName() string { return "sigmoid_mul" }
