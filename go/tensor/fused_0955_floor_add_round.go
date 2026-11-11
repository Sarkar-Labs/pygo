package tensor

import "fmt"

func Fused0955FloorAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Round()
	return r, nil
}
func Fused0955FloorAddRoundName() string { return "floor_add_round" }
