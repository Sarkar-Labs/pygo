package tensor

import "fmt"

func Fused1134RoundMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Ceil()
	return r, nil
}
func Fused1134RoundMatmulCeilName() string { return "round_matmul_ceil" }
