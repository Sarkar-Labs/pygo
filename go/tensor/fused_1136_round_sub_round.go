package tensor

import "fmt"

func Fused1136RoundSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Round()
	return r, nil
}
func Fused1136RoundSubRoundName() string { return "round_sub_round" }
