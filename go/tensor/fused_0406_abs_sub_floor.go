package tensor

import "fmt"

func Fused0406AbsSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Floor()
	return r, nil
}
func Fused0406AbsSubFloorName() string { return "abs_sub_floor" }
