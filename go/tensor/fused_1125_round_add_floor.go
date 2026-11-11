package tensor

import "fmt"

func Fused1125RoundAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Floor()
	return r, nil
}
func Fused1125RoundAddFloorName() string { return "round_add_floor" }
