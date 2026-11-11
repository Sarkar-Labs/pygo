package tensor

import "fmt"

func Fused0174ReluMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Log2()
	return r, nil
}
func Fused0174ReluMatmulLog2Name() string { return "relu_matmul_log2" }
