package tensor

import "fmt"

func Fused0614ExpMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.SiLU()
	return r, nil
}
func Fused0614ExpMatmulSiluName() string { return "exp_matmul_silu" }
