package tensor

import "fmt"

func Fused1169RoundMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Log10()
	return r, nil
}
func Fused1169RoundMatmulLog10Name() string { return "round_matmul_log10" }
