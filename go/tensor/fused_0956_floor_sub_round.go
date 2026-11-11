package tensor

import "fmt"

func Fused0956FloorSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Round()
	return r, nil
}
func Fused0956FloorSubRoundName() string { return "floor_sub_round" }
