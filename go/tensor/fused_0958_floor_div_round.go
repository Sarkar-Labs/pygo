package tensor

import "fmt"

func Fused0958FloorDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Round()
	return r, nil
}
func Fused0958FloorDivRoundName() string { return "floor_div_round" }
