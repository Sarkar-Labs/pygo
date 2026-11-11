package tensor

import "fmt"

func Fused1036CeilSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Floor()
	return r, nil
}
func Fused1036CeilSubFloorName() string { return "ceil_sub_floor" }
