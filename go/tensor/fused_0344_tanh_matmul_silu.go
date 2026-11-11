package tensor

import "fmt"

func Fused0344TanhMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.SiLU()
	return r, nil
}
func Fused0344TanhMatmulSiluName() string { return "tanh_matmul_silu" }
