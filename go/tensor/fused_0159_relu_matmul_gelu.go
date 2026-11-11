package tensor

import "fmt"

func Fused0159ReluMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.GELU()
	return r, nil
}
func Fused0159ReluMatmulGeluName() string { return "relu_matmul_gelu" }
