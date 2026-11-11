package tensor

import "fmt"

func Fused0765SinAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Sin()
	r = r.Floor()
	return r, nil
}
func Fused0765SinAddFloorName() string { return "sin_add_floor" }
