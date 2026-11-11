package tensor

import "fmt"

func Fused0217SigmoidMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Sin()
	return r, nil
}
func Fused0217SigmoidMulSinName() string { return "sigmoid_mul_sin" }
