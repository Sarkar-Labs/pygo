package tensor

import "fmt"

func Fused0946FloorSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Floor()
	return r, nil
}
func Fused0946FloorSubFloorName() string { return "floor_sub_floor" }
