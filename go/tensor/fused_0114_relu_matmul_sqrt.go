package tensor

import "fmt"

func Fused0114ReluMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Sqrt()
	return r, nil
}
func Fused0114ReluMatmulSqrtName() string { return "relu_matmul_sqrt" }
