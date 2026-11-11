package tensor

import "fmt"

func Fused1274GeluMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Tanh()
	return r, nil
}
func Fused1274GeluMatmulTanhName() string { return "gelu_matmul_tanh" }
