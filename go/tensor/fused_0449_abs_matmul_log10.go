package tensor

import "fmt"

func Fused0449AbsMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Log10()
	return r, nil
}
func Fused0449AbsMatmulLog10Name() string { return "abs_matmul_log10" }
