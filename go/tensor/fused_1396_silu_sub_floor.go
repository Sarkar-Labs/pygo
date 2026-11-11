package tensor

import "fmt"

func Fused1396SiluSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Floor()
	return r, nil
}
func Fused1396SiluSubFloorName() string { return "silu_sub_floor" }
