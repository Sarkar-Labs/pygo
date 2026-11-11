package tensor

import "fmt"

func Fused0434AbsMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.SiLU()
	return r, nil
}
func Fused0434AbsMatmulSiluName() string { return "abs_matmul_silu" }
