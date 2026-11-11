package tensor

import "fmt"

func Fused0719LogMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log()
	r = r.Log10()
	return r, nil
}
func Fused0719LogMatmulLog10Name() string { return "log_matmul_log10" }
