package tensor

import "fmt"

func Fused0794SinMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.SiLU()
	return r, nil
}
func Fused0794SinMatmulSiluName() string { return "sin_matmul_silu" }
