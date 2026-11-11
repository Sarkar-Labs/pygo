package tensor

import "fmt"

func Fused0284TanhMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Tanh()
	return r, nil
}
func Fused0284TanhMatmulTanhName() string { return "tanh_matmul_tanh" }
