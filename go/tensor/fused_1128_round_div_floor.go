package tensor

import "fmt"

func Fused1128RoundDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Floor()
	return r, nil
}
func Fused1128RoundDivFloorName() string { return "round_div_floor" }
