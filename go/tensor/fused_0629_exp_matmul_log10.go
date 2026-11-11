package tensor

import "fmt"

func Fused0629ExpMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Log10()
	return r, nil
}
func Fused0629ExpMatmulLog10Name() string { return "exp_matmul_log10" }
