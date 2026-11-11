package tensor

import "fmt"

func Fused1576Log2SubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Floor()
	return r, nil
}
func Fused1576Log2SubFloorName() string { return "log2_sub_floor" }
