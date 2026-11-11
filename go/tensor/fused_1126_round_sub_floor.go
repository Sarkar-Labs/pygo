package tensor

import "fmt"

func Fused1126RoundSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Floor()
	return r, nil
}
func Fused1126RoundSubFloorName() string { return "round_sub_floor" }
