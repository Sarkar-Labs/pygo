package tensor

import "fmt"

func Fused0315TanhAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Floor()
	return r, nil
}
func Fused0315TanhAddFloorName() string { return "tanh_add_floor" }
