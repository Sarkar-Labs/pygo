package tensor

import "fmt"

func Fused0104ReluMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Tanh()
	return r, nil
}
func Fused0104ReluMatmulTanhName() string { return "relu_matmul_tanh" }
