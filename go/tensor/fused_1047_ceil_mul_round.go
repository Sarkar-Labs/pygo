package tensor

import "fmt"

func Fused1047CeilMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Round()
	return r, nil
}
func Fused1047CeilMulRoundName() string { return "ceil_mul_round" }
