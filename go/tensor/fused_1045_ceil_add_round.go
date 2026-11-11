package tensor

import "fmt"

func Fused1045CeilAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Round()
	return r, nil
}
func Fused1045CeilAddRoundName() string { return "ceil_add_round" }
