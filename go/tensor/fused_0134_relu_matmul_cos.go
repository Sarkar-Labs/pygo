package tensor

import "fmt"

func Fused0134ReluMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Cos()
	return r, nil
}
func Fused0134ReluMatmulCosName() string { return "relu_matmul_cos" }
