package tensor

import "fmt"

func Fused0179ReluMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Log10()
	return r, nil
}
func Fused0179ReluMatmulLog10Name() string { return "relu_matmul_log10" }
