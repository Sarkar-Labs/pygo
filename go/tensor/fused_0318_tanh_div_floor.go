package tensor

import "fmt"

func Fused0318TanhDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Tanh()
	r = r.Floor()
	return r, nil
}
func Fused0318TanhDivFloorName() string { return "tanh_div_floor" }
