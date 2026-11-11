package tensor

import "fmt"

func Fused0948FloorDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Floor()
	return r, nil
}
func Fused0948FloorDivFloorName() string { return "floor_div_floor" }
