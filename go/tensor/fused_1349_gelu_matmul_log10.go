package tensor

import "fmt"

func Fused1349GeluMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Log10()
	return r, nil
}
func Fused1349GeluMatmulLog10Name() string { return "gelu_matmul_log10" }
