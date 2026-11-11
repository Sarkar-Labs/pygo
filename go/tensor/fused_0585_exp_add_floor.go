package tensor

import "fmt"

func Fused0585ExpAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_floor: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Floor()
	return r, nil
}
func Fused0585ExpAddFloorName() string { return "exp_add_floor" }
