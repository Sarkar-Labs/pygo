package tensor

import "fmt"

func Fused0164ReluMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.SiLU()
	return r, nil
}
func Fused0164ReluMatmulSiluName() string { return "relu_matmul_silu" }
