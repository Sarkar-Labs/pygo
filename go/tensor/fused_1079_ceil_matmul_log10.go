package tensor

import "fmt"

func Fused1079CeilMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Log10()
	return r, nil
}
func Fused1079CeilMatmulLog10Name() string { return "ceil_matmul_log10" }
