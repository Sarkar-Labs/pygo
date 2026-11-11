package tensor

import "fmt"

func Fused1038CeilDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Floor()
	return r, nil
}
func Fused1038CeilDivFloorName() string { return "ceil_div_floor" }
