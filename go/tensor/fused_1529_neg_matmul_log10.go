package tensor

import "fmt"

func Fused1529NegMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Log10()
	return r, nil
}
func Fused1529NegMatmulLog10Name() string { return "neg_matmul_log10" }
