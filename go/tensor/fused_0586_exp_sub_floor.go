package tensor

import "fmt"

func Fused0586ExpSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_floor: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Floor()
	return r, nil
}
func Fused0586ExpSubFloorName() string { return "exp_sub_floor" }
