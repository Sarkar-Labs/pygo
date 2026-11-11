package tensor

import "fmt"

func Fused0127ReluMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Sin()
	return r, nil
}
func Fused0127ReluMulSinName() string { return "relu_mul_sin" }
