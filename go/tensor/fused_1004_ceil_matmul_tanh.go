package tensor

import "fmt"

func Fused1004CeilMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Tanh()
	return r, nil
}
func Fused1004CeilMatmulTanhName() string { return "ceil_matmul_tanh" }
