package tensor

import "fmt"

func Fused0192SigmoidMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_tanh: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Tanh()
	return r, nil
}
func Fused0192SigmoidMulTanhName() string { return "sigmoid_mul_tanh" }
