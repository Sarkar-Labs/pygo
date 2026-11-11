package tensor

import "fmt"

func Fused1398SiluDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Floor()
	return r, nil
}
func Fused1398SiluDivFloorName() string { return "silu_div_floor" }
