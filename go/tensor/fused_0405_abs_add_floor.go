package tensor

import "fmt"

func Fused0405AbsAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.Floor()
	return r, nil
}
func Fused0405AbsAddFloorName() string { return "abs_add_floor" }
