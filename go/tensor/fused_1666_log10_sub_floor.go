package tensor

import "fmt"

func Fused1666Log10SubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Floor()
	return r, nil
}
func Fused1666Log10SubFloorName() string { return "log10_sub_floor" }
