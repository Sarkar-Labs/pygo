package tensor

import "fmt"

func Fused1575Log2AddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Floor()
	return r, nil
}
func Fused1575Log2AddFloorName() string { return "log2_add_floor" }
