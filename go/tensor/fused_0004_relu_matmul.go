package tensor

import "fmt"

func Fused0004ReluMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	return r, nil
}
func Fused0004ReluMatmulName() string { return "relu_matmul" }
