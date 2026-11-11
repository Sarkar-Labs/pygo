package tensor

import "fmt"

func Fused1046CeilSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Round()
	return r, nil
}
func Fused1046CeilSubRoundName() string { return "ceil_sub_round" }
