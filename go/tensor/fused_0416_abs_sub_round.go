package tensor

import "fmt"

func Fused0416AbsSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Round()
	return r, nil
}
func Fused0416AbsSubRoundName() string { return "abs_sub_round" }
