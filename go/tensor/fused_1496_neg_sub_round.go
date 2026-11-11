package tensor

import "fmt"

func Fused1496NegSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Round()
	return r, nil
}
func Fused1496NegSubRoundName() string { return "neg_sub_round" }
