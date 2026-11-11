package tensor

import "fmt"

func Fused1709Log10MatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Log10()
	return r, nil
}
func Fused1709Log10MatmulLog10Name() string { return "log10_matmul_log10" }
