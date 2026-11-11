package tensor

import "fmt"

func Fused1419SiluMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.GELU()
	return r, nil
}
func Fused1419SiluMatmulGeluName() string { return "silu_matmul_gelu" }
