package tensor

import "fmt"

func Fused0588ExpDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Floor()
	return r, nil
}
func Fused0588ExpDivFloorName() string { return "exp_div_floor" }
