package tensor

import "fmt"

func Fused0144ReluMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Ceil()
	return r, nil
}
func Fused0144ReluMatmulCeilName() string { return "relu_matmul_ceil" }
