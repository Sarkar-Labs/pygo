package tensor

import "fmt"

func Fused1369SiluMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Abs()
	return r, nil
}
func Fused1369SiluMatmulAbsName() string { return "silu_matmul_abs" }
