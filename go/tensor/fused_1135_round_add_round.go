package tensor

import "fmt"

func Fused1135RoundAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Round()
	return r, nil
}
func Fused1135RoundAddRoundName() string { return "round_add_round" }
