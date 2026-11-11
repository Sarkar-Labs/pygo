package tensor

import "fmt"

func Fused1499NegMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_round: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Round()
	return r, nil
}
func Fused1499NegMatmulRoundName() string { return "neg_matmul_round" }
