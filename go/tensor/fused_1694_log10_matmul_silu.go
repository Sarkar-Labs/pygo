package tensor

import "fmt"

func Fused1694Log10MatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_silu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.SiLU()
	return r, nil
}
func Fused1694Log10MatmulSiluName() string { return "log10_matmul_silu" }
