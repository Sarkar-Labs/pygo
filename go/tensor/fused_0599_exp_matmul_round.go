package tensor

import "fmt"

func Fused0599ExpMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Round()
	return r, nil
}
func Fused0599ExpMatmulRoundName() string { return "exp_matmul_round" }
