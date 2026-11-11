package tensor

import "fmt"

func Fused1035CeilAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	r = r.Floor()
	return r, nil
}
func Fused1035CeilAddFloorName() string { return "ceil_add_floor" }
