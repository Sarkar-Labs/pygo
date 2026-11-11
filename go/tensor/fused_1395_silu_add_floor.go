package tensor

import "fmt"

func Fused1395SiluAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Floor()
	return r, nil
}
func Fused1395SiluAddFloorName() string { return "silu_add_floor" }
