package tensor

import "fmt"

func Fused0768SinDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Floor()
	return r, nil
}
func Fused0768SinDivFloorName() string { return "sin_div_floor" }
