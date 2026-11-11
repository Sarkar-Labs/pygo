package tensor

import "fmt"

func Fused0129ReluMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Sin()
	return r, nil
}
func Fused0129ReluMatmulSinName() string { return "relu_matmul_sin" }
