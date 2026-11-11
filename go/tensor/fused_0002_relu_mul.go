package tensor

import "fmt"

func Fused0002ReluMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	return r, nil
}
func Fused0002ReluMulName() string { return "relu_mul" }
