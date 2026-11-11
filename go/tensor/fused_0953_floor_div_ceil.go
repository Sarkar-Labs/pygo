package tensor

import "fmt"

func Fused0953FloorDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Ceil()
	return r, nil
}
func Fused0953FloorDivCeilName() string { return "floor_div_ceil" }
