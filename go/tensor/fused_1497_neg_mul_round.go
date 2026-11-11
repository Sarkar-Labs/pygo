package tensor

import "fmt"

func Fused1497NegMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_round: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Round()
	return r, nil
}
func Fused1497NegMulRoundName() string { return "neg_mul_round" }
