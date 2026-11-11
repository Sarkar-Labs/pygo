package tensor

import "fmt"

func Fused1404SiluMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Ceil()
	return r, nil
}
func Fused1404SiluMatmulCeilName() string { return "silu_matmul_ceil" }
