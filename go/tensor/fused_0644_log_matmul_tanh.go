package tensor

import "fmt"

func Fused0644LogMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Tanh()
	return r, nil
}
func Fused0644LogMatmulTanhName() string { return "log_matmul_tanh" }
