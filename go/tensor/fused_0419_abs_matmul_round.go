package tensor

import "fmt"

func Fused0419AbsMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Round()
	return r, nil
}
func Fused0419AbsMatmulRoundName() string { return "abs_matmul_round" }
