package tensor

import "fmt"

func Fused1439SiluMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Log10()
	return r, nil
}
func Fused1439SiluMatmulLog10Name() string { return "silu_matmul_log10" }
