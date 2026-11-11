package tensor

import "fmt"

func Fused1064CeilMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.SiLU()
	return r, nil
}
func Fused1064CeilMatmulSiluName() string { return "ceil_matmul_silu" }
