package tensor

import "fmt"

func Fused1424SiluMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.SiLU()
	return r, nil
}
func Fused1424SiluMatmulSiluName() string { return "silu_matmul_silu" }
