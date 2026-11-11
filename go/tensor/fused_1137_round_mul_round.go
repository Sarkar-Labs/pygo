package tensor

import "fmt"

func Fused1137RoundMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Round()
	r = r.Round()
	return r, nil
}
func Fused1137RoundMulRoundName() string { return "round_mul_round" }
