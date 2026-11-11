package tensor

import "fmt"

func Fused1665Log10AddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Floor()
	return r, nil
}
func Fused1665Log10AddFloorName() string { return "log10_add_floor" }
