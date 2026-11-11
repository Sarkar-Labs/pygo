package tensor

import "fmt"

func Fused0858CosDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Floor()
	return r, nil
}
func Fused0858CosDivFloorName() string { return "cos_div_floor" }
