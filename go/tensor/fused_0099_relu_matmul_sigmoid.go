package tensor

import "fmt"

func Fused0099ReluMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Sigmoid()
	return r, nil
}
func Fused0099ReluMatmulSigmoidName() string { return "relu_matmul_sigmoid" }
