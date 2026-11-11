package tensor

import "fmt"

func Fused0678LogDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Log()
	r = r.Floor()
	return r, nil
}
func Fused0678LogDivFloorName() string { return "log_div_floor" }
