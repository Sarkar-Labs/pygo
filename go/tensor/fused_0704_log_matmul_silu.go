package tensor

import "fmt"

func Fused0704LogMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.SiLU()
	return r, nil
}
func Fused0704LogMatmulSiluName() string { return "log_matmul_silu" }
