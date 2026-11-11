package tensor

import "fmt"

func Fused1139RoundMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Round()
	return r, nil
}
func Fused1139RoundMatmulRoundName() string { return "round_matmul_round" }
