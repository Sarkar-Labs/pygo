package tensor

import "fmt"

func Fused0596ExpSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Round()
	return r, nil
}
func Fused0596ExpSubRoundName() string { return "exp_sub_round" }
