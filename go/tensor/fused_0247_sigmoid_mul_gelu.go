package tensor

import "fmt"

func Fused0247SigmoidMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.GELU()
	return r, nil
}
func Fused0247SigmoidMulGeluName() string { return "sigmoid_mul_gelu" }
