package tensor

import "fmt"

func Fused0112ReluMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Sqrt()
	return r, nil
}
func Fused0112ReluMulSqrtName() string { return "relu_mul_sqrt" }
