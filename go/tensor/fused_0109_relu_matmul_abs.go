package tensor

import "fmt"

func Fused0109ReluMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Abs()
	return r, nil
}
func Fused0109ReluMatmulAbsName() string { return "relu_matmul_abs" }
