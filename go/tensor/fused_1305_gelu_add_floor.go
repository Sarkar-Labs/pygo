package tensor

import "fmt"

func Fused1305GeluAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Floor()
	return r, nil
}
func Fused1305GeluAddFloorName() string { return "gelu_add_floor" }
