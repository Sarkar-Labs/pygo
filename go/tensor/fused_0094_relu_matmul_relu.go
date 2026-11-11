package tensor

import "fmt"

func Fused0094ReluMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.ReLU()
	return r, nil
}
func Fused0094ReluMatmulReluName() string { return "relu_matmul_relu" }
