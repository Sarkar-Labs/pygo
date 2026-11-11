package tensor

import "fmt"

func Fused1094RoundMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Tanh()
	return r, nil
}
func Fused1094RoundMatmulTanhName() string { return "round_matmul_tanh" }
