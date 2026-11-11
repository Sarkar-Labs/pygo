package tensor

import "fmt"

func Fused0766SinSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Floor()
	return r, nil
}
func Fused0766SinSubFloorName() string { return "sin_sub_floor" }
