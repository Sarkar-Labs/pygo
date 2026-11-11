package tensor

import "fmt"

func Fused1364SiluMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Tanh()
	return r, nil
}
func Fused1364SiluMatmulTanhName() string { return "silu_matmul_tanh" }
