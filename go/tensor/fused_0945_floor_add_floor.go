package tensor

import "fmt"

func Fused0945FloorAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Floor()
	return r, nil
}
func Fused0945FloorAddFloorName() string { return "floor_add_floor" }
